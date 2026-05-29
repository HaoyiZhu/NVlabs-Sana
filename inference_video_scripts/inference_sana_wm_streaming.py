# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end streaming SANA-WM inference.

Four-stream chunk-pipelined recipe:

    * **Stage-1 denoise** — chunk-causal distilled student
      (``SanaMSVideoCamCtrlStreaming``) with self-forcing AR sampling
      (4 steps, ``cfg_scale=1``).
    * **Stage-1 KV save** — runs on its own CUDA stream so it overlaps with
      the refiner + decode of the just-finished chunk instead of sitting on
      stage-1's critical path (bit-exact vs. the legacy single-stream path;
      see ``tests/sana_wm/test_split_kv_save_equiv.py``).
    * **Refiner** — chunk-causal LTX-2 with a sliding KV window (canonical
      3-step distilled schedule).
    * **VAE** — causal LTX-2 VAE that decodes one block at a time.

One decoded chunk per AR block is appended to a progressive MP4 you can
watch while generation continues.

The script applies the canonical fast configuration by default — no flags
needed:

    * ``torch.compile`` on the VAE decoder + refiner transformer
      (``max-autotune-no-cudagraphs``, numerically equivalent to eager).
    * Flash-only SDPA, Inductor ``coordinate_descent_tuning`` + ``epilogue_fusion``,
      cuDNN benchmark, expandable CUDA allocator.

Reaches ~0.95× of realtime in steady-state on a single H100 after the
one-time ``torch.compile`` warmup (~3 min cold, ~30 s warm cache).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

# Env knobs that must be set before any ``torch`` / ``diffusion.*`` import.
# DISABLE_XFORMERS keeps cross-attention on plain SDPA with Python-list
# seqlens (the layout the self-forcing scheduler expects). expandable_segments
# lets Inductor's max-autotune explore larger Triton workspaces without
# fragmentation OOMs on the refiner KV window.
os.environ.setdefault("DISABLE_XFORMERS", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import pyrallis  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from diffusion.utils.logger import get_root_logger  # noqa: E402
from inference_video_scripts.inference_sana_wm import (  # noqa: E402
    GenerationParams,
    InferenceConfig,
    RefinerSettings,
    SanaWMPipeline,
    _resolve_trajectory,
    _snap_num_frames,
    action_string_to_c2w,  # noqa: F401  (re-export)
    estimate_intrinsics_with_pi3x,
    load_intrinsics,
    resize_and_center_crop,
    transform_intrinsics_for_crop,
)

# Canonical 4-step distilled-student schedule.
DEFAULT_DENOISING_STEP_LIST = "1000,960,889,727,0"

# Default location of the streaming weights bundle.
DEFAULT_STREAMING_ROOT = Path("pretrained_models/sana_wm_streaming")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end streaming SANA-WM inference (stage-1 + refiner + causal VAE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--image", type=Path, required=True, help="First-frame RGB image.")
    p.add_argument("--prompt", type=Path, required=True, help="UTF-8 text file with the prompt.")
    p.add_argument("--output_dir", type=Path, required=True, help="Directory to write the progressive mp4.")
    p.add_argument("--name", default="output", help="Filename stem for outputs.")

    cam_group = p.add_mutually_exclusive_group(required=True)
    cam_group.add_argument("--camera", type=Path, help="(F,4,4) .npy camera-to-world poses.")
    cam_group.add_argument(
        "--action", type=str, help="Action DSL string, e.g. 'w-120,lw-80,...'. Rolled out internally."
    )

    p.add_argument("--translation_speed", type=float, default=0.04,
                   help="Per-frame translation magnitude when --action is used.")
    p.add_argument("--rotation_speed_deg", type=float, default=1.2,
                   help="Per-frame rotation magnitude (degrees) when --action is used.")
    p.add_argument("--intrinsics", type=Path, default=None,
                   help="(3,3), (F,3,3), or (4,) intrinsics .npy. Pi3X-estimated if omitted.")
    p.add_argument("--num_frames", type=int, default=961,
                   help="Total pixel frames (LTX-2 snaps to 8k+1).")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--flow_shift", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative_prompt", default="")

    # Streaming-only knobs.
    p.add_argument("--streaming_root", type=Path, default=DEFAULT_STREAMING_ROOT,
                   help="Directory holding sana_dit/, ltx2_causal_vae/, refiner_diffusers/, gemma3_12b/, and the yaml.")
    p.add_argument("--config", type=Path, default=None,
                   help="Override the streaming YAML. Default: <streaming_root>/sana_wm_streaming_1600m_720p.yaml.")
    p.add_argument("--model_path", type=Path, default=None,
                   help="Override the streaming DiT checkpoint. Default: <streaming_root>/sana_dit/model.pt.")
    p.add_argument("--causal_vae_path", type=Path, default=None,
                   help="Override the causal LTX-2 VAE directory. Default: <streaming_root>/ltx2_causal_vae.")
    p.add_argument("--refiner_root", type=Path, default=None,
                   help="Override the chunk-causal refiner directory. Default: <streaming_root>/refiner_diffusers.")
    p.add_argument("--refiner_gemma_root", type=Path, default=None,
                   help="Override the Gemma diffusers root. Default: <streaming_root>/gemma3_12b.")
    p.add_argument("--denoising_step_list", default=DEFAULT_DENOISING_STEP_LIST,
                   help="Comma-separated distilled-student timestep schedule (must end with 0).")
    p.add_argument("--num_frame_per_block", type=int, default=3,
                   help="Latent frames per stage-1 AR chunk (must match the model's chunk_size).")
    p.add_argument("--refiner_block_size", type=int, default=3,
                   help="Refiner latent frames per AR block.")
    p.add_argument("--refiner_kv_max_frames", type=int, default=11,
                   help="Refiner KV sliding-window size (sink + history + active).")
    p.add_argument("--refiner_seed", type=int, default=42)
    p.add_argument("--sink_size", type=int, default=1)
    p.add_argument("--no_sink_token", action="store_true",
                   help="Disable the stage-1 sink token (default: enabled).")
    p.add_argument("--num_cached_blocks", type=int, default=2,
                   help="Stage-1 KV sliding-window size (-1 keeps all past chunks).")
    p.add_argument("--streaming_crf", type=int, default=18,
                   help="ffmpeg CRF for the progressive MP4 (lower = higher quality).")
    p.add_argument("--streaming_preset", default="medium",
                   help="ffmpeg libx264 preset for the progressive MP4 writer.")

    p.add_argument("--offload_vae", action="store_true",
                   help="Move the VAE to CPU between encode/decode steps.")
    p.add_argument("--offload_refiner", action="store_true",
                   help="Lazy-load the LTX-2 refiner only when needed; release afterwards.")
    return p


def _resolve_streaming_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    """Materialise the five checkpoint paths from --streaming_root + overrides."""
    root = args.streaming_root
    config_path = args.config or (root / "sana_wm_streaming_1600m_720p.yaml")
    model_path = args.model_path or (root / "sana_dit" / "model.pt")
    causal_vae_path = args.causal_vae_path or (root / "ltx2_causal_vae")
    refiner_root = args.refiner_root or (root / "refiner_diffusers")
    gemma_root = args.refiner_gemma_root or (root / "gemma3_12b")
    for label, path in (
        ("--config", config_path),
        ("--model_path", model_path),
        ("--causal_vae_path", causal_vae_path),
        ("--refiner_root", refiner_root),
        ("--refiner_gemma_root", gemma_root),
    ):
        if not Path(path).exists():
            raise SystemExit(f"{label} does not exist: {path}")
    return config_path, model_path, causal_vae_path, refiner_root, gemma_root


def _apply_fast_defaults() -> None:
    """Set the numerically-neutral perf knobs (cuDNN bench, flash SDPA, Inductor)."""
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(True)
    import torch._inductor.config as _ic
    _ic.coordinate_descent_tuning = True
    _ic.epilogue_fusion = True


def main() -> None:
    args = _build_parser().parse_args()
    logger: logging.Logger = get_root_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _apply_fast_defaults()

    image = Image.open(args.image).convert("RGB")
    prompt = args.prompt.read_text(encoding="utf-8", errors="replace").strip()
    if not prompt:
        raise SystemExit(f"Prompt file is empty: {args.prompt}")

    c2w_full = _resolve_trajectory(args)
    num_frames = min(args.num_frames, c2w_full.shape[0])
    snapped = _snap_num_frames(num_frames, stride=8, upper_bound=c2w_full.shape[0])
    if snapped != args.num_frames:
        logger.warning(
            "LTX-2 VAE requires num_frames = 8k+1; "
            f"--num_frames={args.num_frames} snapped to {snapped} "
            f"(trajectory has {c2w_full.shape[0]} frames)."
        )
    num_frames = snapped
    c2w = c2w_full[:num_frames]

    cropped, src_size, resized_size, crop_offset = resize_and_center_crop(image)
    if args.intrinsics is not None:
        intr_src = load_intrinsics(args.intrinsics, num_frames)
    else:
        intr_one = estimate_intrinsics_with_pi3x(image, device, logger)
        intr_src = np.broadcast_to(intr_one, (num_frames, 4)).copy()
    intrinsics_vec4 = transform_intrinsics_for_crop(intr_src, src_size, resized_size, crop_offset)

    config_path, model_path, causal_vae_path, refiner_root, gemma_root = _resolve_streaming_paths(args)
    config: InferenceConfig = pyrallis.parse(
        config_class=InferenceConfig, config_path=str(config_path), args=[]
    )
    config.vae.vae_type = "LTX2VAE_diffusers_causal"
    config.vae.vae_pretrained = str(causal_vae_path)
    logger.info(f"[causal-vae] vae_pretrained -> {config.vae.vae_pretrained}")

    refiner = RefinerSettings(
        root=str(refiner_root),
        gemma_root=str(gemma_root),
        sink_size=args.sink_size,
        seed=args.refiner_seed,
        block_size=args.refiner_block_size,
        kv_max_frames=args.refiner_kv_max_frames,
    )

    pipeline = SanaWMPipeline(
        config=config,
        model_path=str(model_path),
        device=device,
        refiner=refiner,
        offload_vae=args.offload_vae,
        offload_refiner=args.offload_refiner,
        logger=logger,
    )

    # Numerically-equivalent compile (default Inductor mode, no CUDA-graph
    # capture, no fp32->fp16 substitution). Cold compile takes ~3 min the
    # first time; subsequent runs reuse the Inductor cache.
    logger.info(
        "[streaming] torch.compile(vae.decoder + refiner.transformer, "
        "mode='max-autotune-no-cudagraphs')"
    )
    pipeline.vae.decoder = torch.compile(
        pipeline.vae.decoder, mode="max-autotune-no-cudagraphs", dynamic=True
    )
    pipeline.refiner.transformer = torch.compile(
        pipeline.refiner.transformer, mode="max-autotune-no-cudagraphs", dynamic=True
    )

    denoising_step_list = [int(t.strip()) for t in args.denoising_step_list.split(",") if t.strip()]
    if not denoising_step_list or denoising_step_list[-1] != 0:
        raise SystemExit("--denoising_step_list must be a comma-separated list ending with 0.")

    params = GenerationParams(
        num_frames=num_frames,
        fps=args.fps,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        sampling_algo="self_forcing",
        num_cached_blocks=args.num_cached_blocks,
        sink_token=not args.no_sink_token,
        num_frame_per_block=args.num_frame_per_block,
        denoising_step_list=denoising_step_list,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    streaming_path = out_dir / f"{args.name}_streaming.mp4"
    logger.info(f"[streaming] starting interactive chunk-pipelined inference -> {streaming_path}")
    result = pipeline.generate_streaming(
        cropped,
        prompt,
        c2w,
        intrinsics_vec4,
        params,
        output_path=streaming_path,
        streaming_crf=args.streaming_crf,
        streaming_preset=args.streaming_preset,
    )
    logger.info(
        f"[streaming] done: wrote {result['n_pixel_frames']} frames to {result['output_path']}"
    )


if __name__ == "__main__":
    main()
