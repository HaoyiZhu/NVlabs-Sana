"""Bit-exact equivalence test for ``yield_save_separately`` in ``sample_chunks``.

The streaming orchestrator drives stage-1's KV-save pass on a dedicated CUDA
stream by calling ``sample_chunks(..., yield_save_separately=True)``: the
chunk view is yielded *before* the save and a ``None`` sentinel is yielded
*after* the save. This must not change any output — same seed and inputs
must produce identical per-chunk latents in either yield mode.

Run with:

    python -m tests.sana_wm.test_split_kv_save_equiv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("DISABLE_XFORMERS", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pyrallis  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402

from diffusion.scheduler.self_forcing_flow_euler_sampler import (  # noqa: E402
    SelfForcingFlowEulerCamCtrl,
)
from diffusion.utils.chunk_utils import get_chunk_index_from_config  # noqa: E402
from diffusion.utils.logger import get_root_logger  # noqa: E402
from inference_video_scripts.inference_sana_wm import (  # noqa: E402
    TARGET_HEIGHT,
    TARGET_WIDTH,
    GenerationParams,
    InferenceConfig,
    RefinerSettings,
    SanaWMPipeline,
    action_string_to_c2w,
    load_intrinsics,
    prepare_camera,
    resize_and_center_crop,
    transform_intrinsics_for_crop,
    vae_encode,
)
from inference_video_scripts.inference_sana_wm_streaming import (  # noqa: E402
    DEFAULT_DENOISING_STEP_LIST,
    DEFAULT_STREAMING_ROOT,
    _resolve_streaming_paths,
)


def _build_iter(
    *,
    pipeline: SanaWMPipeline,
    params: GenerationParams,
    cached: dict,
    latent_T: int,
    latent_h: int,
    latent_w: int,
    config: InferenceConfig,
    device: torch.device,
    yield_save_separately: bool,
):
    """Fresh solver + ``sample_chunks`` iterator on a deterministic seed."""
    latent_channels = cached["first_latent"].shape[1]
    generator = torch.Generator(device=device).manual_seed(params.seed)
    z = torch.randn(
        1, latent_channels, latent_T, latent_h, latent_w,
        dtype=pipeline.weight_dtype, device=device, generator=generator,
    )
    z[:, :, :1] = cached["first_latent"]

    chunk_index = get_chunk_index_from_config(config, num_frames=latent_T)
    model_kwargs: dict[str, object] = dict(
        data_info={
            "img_hw": torch.tensor(
                [[TARGET_HEIGHT, TARGET_WIDTH]], dtype=torch.float, device=device
            ),
            "condition_frame_info": {0: 0.0},
        },
        mask=cached["mask_cfg"],
        camera_conditions=cached["raymap_cfg"],
        chunk_plucker=cached["chunk_plucker_cfg"],
    )
    if chunk_index is not None:
        model_kwargs["chunk_index"] = chunk_index

    solver = SelfForcingFlowEulerCamCtrl(
        pipeline.model,
        condition=cached["cond"],
        uncondition=cached["neg"],
        cfg_scale=params.cfg_scale,
        flow_shift=pipeline._resolve_flow_shift(params.flow_shift),
        model_kwargs=model_kwargs,
        base_chunk_frames=params.num_frame_per_block,
        num_cached_blocks=params.num_cached_blocks,
        sink_token=params.sink_token,
        use_softmax_attention=True,
    )
    return solver.sample_chunks(
        z,
        steps=params.step,
        generator=generator,
        denoising_step_list=params.denoising_step_list,
        yield_save_separately=yield_save_separately,
    )


def _drive(
    stage1_iter,
    *,
    yield_save_separately: bool,
    n_chunks: int,
) -> list[tuple[int, int, int, torch.Tensor]]:
    """Exhaust the iterator and snapshot each chunk view for later comparison."""
    chunks: list[tuple[int, int, int, torch.Tensor]] = []
    for _ in range(n_chunks):
        first = next(stage1_iter)
        if first is None:
            raise AssertionError("First yield was None — unexpected.")
        chunk_idx, latent_view, start_f, end_f = first
        # Clone so the snapshot survives subsequent in-place writes to ``latents``.
        chunks.append((chunk_idx, int(start_f), int(end_f), latent_view.detach().clone()))
        if yield_save_separately:
            sentinel = next(stage1_iter)
            if sentinel is not None:
                raise AssertionError(f"Expected None sentinel after KV-save, got {sentinel!r}")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("asset/sana_wm/demo_0.png"))
    parser.add_argument("--prompt", type=Path, default=Path("asset/sana_wm/demo_0.txt"))
    parser.add_argument("--intrinsics", type=Path, default=Path("asset/sana_wm/demo_0_intrinsics.npy"))
    parser.add_argument(
        "--action",
        default="w-80,jw-40,w-40,lw-60,w-100,w-100,jw-40,w-100,lw-60,w-100",
    )
    parser.add_argument("--translation_speed", type=float, default=0.055)
    parser.add_argument("--rotation_speed_deg", type=float, default=1.2)
    parser.add_argument("--num_frames", type=int, default=241)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming_root", type=Path, default=DEFAULT_STREAMING_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--causal_vae_path", type=Path, default=None)
    parser.add_argument("--refiner_root", type=Path, default=None)
    parser.add_argument("--refiner_gemma_root", type=Path, default=None)
    parser.add_argument("--num_frame_per_block", type=int, default=3)
    parser.add_argument("--sink_size", type=int, default=1)
    parser.add_argument(
        "--bf16_noise_atol",
        type=float,
        default=1e-3,
        help="Maximum acceptable absolute divergence when the two paths are not "
             "bit-exact. Defaults to bf16 round-off tolerance.",
    )
    args = parser.parse_args()

    logger = get_root_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required.")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(True)

    # Inputs / pipeline — mirrors the production launch path; intrinsics come
    # from the precomputed .npy so the test doesn't depend on Pi3X.
    from inference_video_scripts.inference_sana_wm import _snap_num_frames

    image = Image.open(args.image).convert("RGB")
    prompt = args.prompt.read_text(encoding="utf-8", errors="replace").strip()
    c2w_full = action_string_to_c2w(
        args.action,
        translation_speed=args.translation_speed,
        rotation_speed_deg=args.rotation_speed_deg,
    )
    num_frames = min(args.num_frames, c2w_full.shape[0])
    snapped = _snap_num_frames(num_frames, stride=8, upper_bound=c2w_full.shape[0])
    if snapped != args.num_frames:
        logger.warning(f"snapped {args.num_frames} -> {snapped}")
    num_frames = snapped
    c2w = c2w_full[:num_frames]

    cropped, src_size, resized_size, crop_offset = resize_and_center_crop(image)
    intr_src = load_intrinsics(args.intrinsics, num_frames)
    intrinsics_vec4 = transform_intrinsics_for_crop(intr_src, src_size, resized_size, crop_offset)

    config_path, model_path, causal_vae_path, refiner_root, gemma_root = _resolve_streaming_paths(args)
    config: InferenceConfig = pyrallis.parse(
        config_class=InferenceConfig, config_path=str(config_path), args=[]
    )
    config.vae.vae_type = "LTX2VAE_diffusers_causal"
    config.vae.vae_pretrained = str(causal_vae_path)

    refiner_settings = RefinerSettings(
        root=str(refiner_root),
        gemma_root=str(gemma_root),
        sink_size=args.sink_size,
        seed=42,
    )
    pipeline = SanaWMPipeline(
        config=config,
        model_path=str(model_path),
        device=device,
        refiner=refiner_settings,
        offload_vae=False,
        offload_refiner=False,
        logger=logger,
    )

    denoising_step_list = [int(t.strip()) for t in DEFAULT_DENOISING_STEP_LIST.split(",") if t.strip()]
    params = GenerationParams(
        num_frames=num_frames,
        fps=16,
        cfg_scale=1.0,
        flow_shift=8.0,
        seed=args.seed,
        negative_prompt="",
        sampling_algo="self_forcing",
        num_cached_blocks=2,
        sink_token=True,
        num_frame_per_block=args.num_frame_per_block,
        denoising_step_list=denoising_step_list,
    )

    vae_stride = config.vae.vae_stride
    latent_T = (params.num_frames - 1) // vae_stride[0] + 1
    latent_h = TARGET_HEIGHT // vae_stride[-1]
    latent_w = TARGET_WIDTH // vae_stride[-1]
    camera = prepare_camera(
        c2w[: params.num_frames],
        intrinsics_vec4[: params.num_frames],
        target_size=(TARGET_HEIGHT, TARGET_WIDTH),
        vae_stride=vae_stride,
    )
    img = (T.ToTensor()(cropped) * 2.0 - 1.0).unsqueeze(0).unsqueeze(2)
    first_latent = vae_encode(
        config.vae.vae_type, pipeline.vae,
        img.to(device, dtype=pipeline.vae_dtype), device=device,
    ).to(pipeline.weight_dtype)
    cond, cond_mask, neg, _neg_mask = pipeline._encode_prompts(prompt, "")
    raymap = camera["raymap"].unsqueeze(0).to(device, dtype=pipeline.weight_dtype)
    chunk_plucker = camera["chunk_plucker"].unsqueeze(0).to(device, dtype=pipeline.weight_dtype)
    cached = {
        "first_latent": first_latent,
        "cond": cond,
        "neg": neg,
        "mask_cfg": cond_mask,
        "raymap_cfg": raymap,
        "chunk_plucker_cfg": chunk_plucker,
    }

    n_chunks = latent_T // params.num_frame_per_block
    build_kwargs = dict(
        pipeline=pipeline, params=params, cached=cached,
        latent_T=latent_T, latent_h=latent_h, latent_w=latent_w,
        config=config, device=device,
    )

    logger.info("[equiv] RUN A: yield_save_separately=False (legacy single yield)")
    chunks_a = _drive(
        _build_iter(**build_kwargs, yield_save_separately=False),
        yield_save_separately=False, n_chunks=n_chunks,
    )
    torch.cuda.synchronize()

    logger.info("[equiv] RUN B: yield_save_separately=True (split yield)")
    chunks_b = _drive(
        _build_iter(**build_kwargs, yield_save_separately=True),
        yield_save_separately=True, n_chunks=n_chunks,
    )
    torch.cuda.synchronize()

    logger.info(f"[equiv] comparing {len(chunks_a)} chunks")
    assert len(chunks_a) == len(chunks_b), (len(chunks_a), len(chunks_b))
    max_abs = 0.0
    max_rel = 0.0
    for (ia, sa, ea, ta), (ib, sb, eb, tb) in zip(chunks_a, chunks_b):
        assert (ia, sa, ea) == (ib, sb, eb), ((ia, sa, ea), (ib, sb, eb))
        if torch.equal(ta, tb):
            continue
        diff = (ta.float() - tb.float()).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        denom = tb.float().abs().clamp_min(1e-9)
        max_rel = max(max_rel, float((diff / denom).max().item()))

    if max_abs == 0.0:
        logger.info("[equiv] PASSED (bit-exact)")
        return
    if max_abs <= args.bf16_noise_atol:
        logger.info(
            f"[equiv] PASSED within bf16 noise (max_abs={max_abs:.3e}, max_rel={max_rel:.3e})"
        )
        return
    raise SystemExit(
        f"FAILED: split-save introduced non-trivial divergence "
        f"(max_abs={max_abs:.3e}, max_rel={max_rel:.3e})"
    )


if __name__ == "__main__":
    main()
