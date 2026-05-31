#!/usr/bin/env python3
"""Realtime Gradio frontend for the optimized Sana-WM streaming path."""

from __future__ import annotations

import argparse
import html
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# These must be set before importing torch/diffusion modules.
os.environ.setdefault("DISABLE_XFORMERS", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if "--cuda_visible_devices" in sys.argv:
    idx = sys.argv.index("--cuda_visible_devices")
    if idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[idx + 1]

# Same default candidate as scripts/benchmark_sana_wm_5090_realtime.sh.
os.environ.setdefault("DPM_TQDM", "True")
os.environ.setdefault("FUSED_GDN_PRECISION", "2")
os.environ.setdefault("SANA_WM_TORCH_COMPILE_DYNAMIC", "0")
os.environ.setdefault("SANA_WM_TORCH_COMPILE_TARGETS", "vae")
os.environ.setdefault("SANA_WM_STAGE1_KV_SAVE_STRIDE", "0")
os.environ.setdefault("SANA_WM_STAGE1_LINEARIZE_FFN", "1")
os.environ.setdefault("SANA_WM_STAGE1_NVFP4", "1")
os.environ.setdefault("SANA_WM_STAGE1_NVFP4_MODE", "self_attn+cross+ffn")
os.environ.setdefault("SANA_WM_STAGE1_NVFP4_TEXT_PAD_MULTIPLE", "8")
os.environ.setdefault("SANA_WM_SDPA_D112_DIRECT", "1")
os.environ.setdefault("SANA_WM_REFINER_ATTN_BACKEND", "_native_flash")
os.environ.setdefault("SANA_WM_REFINER_SELF_ATTN_KERNEL", "flash_attn")
os.environ.setdefault("SANA_WM_REFINER_CROSS_ATTN_KV_CACHE", "1")
os.environ.setdefault("SANA_WM_REFINER_EMPTY_CACHE_BEFORE_PREFIX", "0")
os.environ.setdefault("SANA_WM_REFINER_EMPTY_CACHE_BEFORE_CAPTURE", "0")
os.environ.setdefault("SANA_WM_REFINER_KV_CACHE_DTYPE", "fp8_e4m3fn")
os.environ.setdefault("SANA_WM_REFINER_NVFP4", "1")
os.environ.setdefault("SANA_WM_REFINER_PRECONCAT_PREFIX", "1")
os.environ.setdefault("SANA_WM_REFINER_NO_CLONE_CAPTURED_KV", "1")
os.environ.setdefault("SANA_WM_REFINER_CAPTURE_KV_ONLY_LAST", "1")
os.environ.setdefault("SANA_WM_REFINER_FAST_KV_CAPTURE", "last_predict")
os.environ.setdefault("SANA_WM_REFINER_FAST_KV_CLEAN_INTERVAL", "4")
os.environ.setdefault("SANA_WM_STREAMING_PREDECODE_SINK", "1")
os.environ.setdefault("SANA_WM_STREAMING_LAZY_VAE_DECODER", "1")
os.environ.setdefault("SANA_WM_STREAMING_PROMPT_CACHE", "1")
os.environ.setdefault("SANA_WM_TE_NVFP4_CPU_STAGING", "1")

import gradio as gr  # noqa: E402
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
    _snap_num_frames,
    action_string_to_c2w,
    load_intrinsics,
    resize_and_center_crop,
    transform_intrinsics_for_crop,
)
from inference_video_scripts.inference_sana_wm_streaming import (  # noqa: E402
    DEFAULT_DENOISING_STEP_LIST,
    DEFAULT_STREAMING_ROOT,
    _apply_fast_defaults,
)
from inference_video_scripts.streaming_mp4_writer import resolve_ffmpeg_exe  # noqa: E402


TARGET_HEIGHT = 704
TARGET_WIDTH = 1280
DEFAULT_ACTION = "w-240,jw-120,w-240,lw-120,w-240"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--streaming_root", type=Path, default=DEFAULT_STREAMING_ROOT)
    parser.add_argument("--output_dir", type=Path, default=Path("demo_outputs/sana_wm_realtime"))
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--lazy_load", action="store_true")
    return parser


def _ffmpeg_has_encoder(encoder: str) -> bool:
    encoder = str(encoder).strip().lower()
    try:
        import subprocess

        out = subprocess.run(
            [resolve_ffmpeg_exe(), "-hide_banner", "-encoders"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return False
    except RuntimeError:
        return False
    return encoder in out


def _resolve_streaming_paths(streaming_root: Path) -> tuple[Path, Path, Path, Path, Path]:
    config_path = streaming_root / "sana_wm_streaming_1600m_720p.yaml"
    model_path = streaming_root / "sana_dit" / "model.pt"
    causal_vae_path = streaming_root / "ltx2_causal_vae"
    refiner_root = streaming_root / "refiner_diffusers"
    gemma_root = streaming_root / "gemma3_12b"
    for label, path in (
        ("config", config_path),
        ("model", model_path),
        ("causal VAE", causal_vae_path),
        ("refiner", refiner_root),
        ("Gemma", gemma_root),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} path does not exist: {path}")
    return config_path, model_path, causal_vae_path, refiner_root, gemma_root


def _default_prompt() -> str:
    path = Path("asset/sana_wm/demo_0.txt")
    return path.read_text(encoding="utf-8", errors="replace").strip() if path.exists() else ""


class DemoState:
    def __init__(self, *, streaming_root: Path, no_compile: bool) -> None:
        self.streaming_root = streaming_root
        self.no_compile = no_compile
        self.lock = threading.Lock()
        self.pipeline: SanaWMPipeline | None = None

    def get_pipeline(self) -> SanaWMPipeline:
        with self.lock:
            if self.pipeline is not None:
                return self.pipeline
            logger = get_root_logger()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _apply_fast_defaults()
            config_path, model_path, causal_vae_path, refiner_root, gemma_root = _resolve_streaming_paths(
                self.streaming_root
            )
            config: InferenceConfig = pyrallis.parse(
                config_class=InferenceConfig, config_path=str(config_path), args=[]
            )
            config.vae.vae_type = "LTX2VAE_diffusers_causal"
            config.vae.vae_pretrained = str(causal_vae_path)
            refiner = RefinerSettings(
                root=str(refiner_root),
                gemma_root=str(gemma_root),
                sink_size=1,
                seed=42,
                block_size=3,
                kv_max_frames=2,
            )
            self.pipeline = SanaWMPipeline(
                config=config,
                model_path=str(model_path),
                device=device,
                refiner=refiner,
                offload_vae=False,
                offload_refiner=False,
                offload_text_encoder=False,
                logger=logger,
            )
            if not self.no_compile:
                compile_mode = os.environ.get("SANA_WM_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs")
                dynamic = os.environ.get("SANA_WM_TORCH_COMPILE_DYNAMIC", "0").lower() not in {
                    "0",
                    "false",
                    "no",
                    "off",
                }
                targets = {
                    item.strip().lower()
                    for item in os.environ.get("SANA_WM_TORCH_COMPILE_TARGETS", "vae").split(",")
                    if item.strip()
                }
                if "vae" in targets:
                    self.pipeline.vae.decoder = torch.compile(
                        self.pipeline.vae.decoder,
                        mode=compile_mode,
                        dynamic=dynamic,
                    )
                if "refiner" in targets:
                    self.pipeline.refiner.transformer = torch.compile(
                        self.pipeline.refiner.transformer,
                        mode=compile_mode,
                        dynamic=dynamic,
                    )
            return self.pipeline


def _make_preview(frame: np.ndarray) -> Image.Image:
    image = Image.fromarray(frame)
    return image.resize((640, 352), Image.Resampling.LANCZOS)


def _coerce_image(image: Image.Image | np.ndarray | str | None) -> Image.Image:
    if image is None:
        image = Image.open("asset/sana_wm/demo_0.png")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str):
        image = Image.open(image)
    return image.convert("RGB")


def _intrinsics_path(uploaded) -> Path:
    if uploaded is None:
        return Path("asset/sana_wm/demo_0_intrinsics.npy")
    name = getattr(uploaded, "name", None)
    return Path(name or uploaded)


def _progress_html(
    *,
    save_mp4: bool,
    encoder: str,
    message: str,
    frames: int = 0,
    done: int = 0,
    total: int = 0,
    elapsed: float = 0.0,
    complete: bool = False,
    path: str | None = None,
) -> str:
    total = max(0, int(total))
    done = max(0, int(done))
    pct = 100.0 if complete else (100.0 * done / total if total > 0 else 0.0)
    pct = max(0.0, min(100.0, pct))
    mode = "MP4" if save_mp4 else "Preview only"
    safe_message = html.escape(str(message))
    safe_encoder = html.escape(str(encoder))
    safe_path = html.escape(str(path or ""))
    path_line = f"<div class='mp4-progress-path'>{safe_path}</div>" if path else ""
    return f"""
    <div class="mp4-progress">
      <div class="mp4-progress-top">
        <span>{mode} progress</span>
        <span>{pct:.1f}%</span>
      </div>
      <div class="mp4-progress-bar"><div style="width:{pct:.2f}%"></div></div>
      <div class="mp4-progress-meta">{safe_message} | chunks {done}/{total or '?'} | frames {int(frames)} | {elapsed:.1f}s | {safe_encoder}</div>
      {path_line}
    </div>
    """


def _preload_pipeline(state: DemoState) -> None:
    t0 = time.perf_counter()
    print("[demo] preloading Sana-WM pipeline before Gradio launch", flush=True)
    pipeline = state.get_pipeline()
    prompt = _default_prompt()
    if prompt and os.environ.get("SANA_WM_DEMO_PREWARM_PROMPT", "1").lower() not in {"0", "false", "no", "off"}:
        print("[demo] pre-encoding default stage-1/refiner prompts", flush=True)
        with torch.inference_mode():
            pipeline._get_streaming_refiner_prompt(prompt)
            pipeline._get_streaming_stage1_prompt(prompt, "")
    print("[demo] preparing resident NVFP4/streaming modules", flush=True)
    with torch.inference_mode():
        cpu_stage_nvfp4 = os.environ.get("SANA_WM_TE_NVFP4_CPU_STAGING", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if cpu_stage_nvfp4:
            pipeline.refiner.offload_video_unused_audio_modules("cpu")
            pipeline.refiner.prepare_transformer_nvfp4()
            pipeline.refiner.move_video_modules(pipeline.device)
            pipeline.refiner.offload_video_unused_audio_modules("cpu")
        else:
            pipeline.refiner.move_video_modules(pipeline.device)
            pipeline.refiner.offload_video_unused_audio_modules("cpu")
            pipeline.refiner.prepare_transformer_nvfp4()
        pipeline.model.to(pipeline.device)
        pipeline._prepare_stage1_nvfp4()
        if os.environ.get("SANA_WM_STREAMING_LAZY_VAE_DECODER", "1").lower() in {"1", "true", "yes", "on"}:
            pipeline._move_vae_decoder_for_streaming("cpu")
        else:
            pipeline._move_vae_decoder_for_streaming(pipeline.device)
        torch.cuda.empty_cache()
    print(f"[demo] pipeline preload complete in {time.perf_counter() - t0:.1f}s", flush=True)


def build_demo(args: argparse.Namespace, state: DemoState) -> gr.Blocks:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        image,
        prompt: str,
        action: str,
        num_frames: int,
        translation_speed: float,
        rotation_speed_deg: float,
        seed: int,
        save_mp4: bool,
        encoder_choice: str,
        intrinsics_file,
    ):
        updates: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=64)
        started = time.perf_counter()
        selected_encoder = "h264_nvenc" if encoder_choice == "h264_nvenc" else "libx264"
        output_mode = "mp4" if save_mp4 else "cpu"
        if save_mp4:
            if selected_encoder == "h264_nvenc" and not _ffmpeg_has_encoder("h264_nvenc"):
                updates.put(("status", "h264_nvenc is not available in ffmpeg; falling back to libx264."))
                selected_encoder = "libx264"
            if not _ffmpeg_has_encoder(selected_encoder):
                updates.put(("status", "ffmpeg is not available; preview will stream without final MP4."))
                output_mode = "cpu"
        else:
            updates.put(("status", "MP4 saving is off; streaming decoded chunks directly to the preview."))

        def put_update(kind: str, value: object) -> None:
            try:
                updates.put_nowait((kind, value))
            except queue.Full:
                try:
                    updates.get_nowait()
                except queue.Empty:
                    pass
                updates.put_nowait((kind, value))

        def on_chunk(pixel_np: np.ndarray, frame_base: int, chunk_idx: int) -> None:
            if pixel_np.shape[0] <= 0:
                return
            put_update(
                "frame",
                (
                    pixel_np[-1].copy(),
                    f"chunk {chunk_idx:02d}  frames {frame_base}-{frame_base + pixel_np.shape[0] - 1}",
                ),
            )

        def on_progress(event: dict[str, object]) -> None:
            put_update("progress", dict(event))

        def worker() -> None:
            try:
                pipeline = state.get_pipeline()
                put_update("status", "pipeline ready; preparing request")
                pil_image = _coerce_image(image)
                prompt_text = (prompt or "").strip()
                put_update("status", "preparing camera path and intrinsics")
                c2w_full = action_string_to_c2w(
                    action or DEFAULT_ACTION,
                    translation_speed=float(translation_speed),
                    rotation_speed_deg=float(rotation_speed_deg),
                )
                requested = min(int(num_frames), int(c2w_full.shape[0]))
                snapped = _snap_num_frames(requested, stride=8, upper_bound=int(c2w_full.shape[0]))
                c2w = c2w_full[:snapped]
                cropped, src_size, resized_size, crop_offset = resize_and_center_crop(pil_image)
                intr_src = load_intrinsics(_intrinsics_path(intrinsics_file), snapped)
                intrinsics_vec4 = transform_intrinsics_for_crop(intr_src, src_size, resized_size, crop_offset)
                params = GenerationParams(
                    num_frames=snapped,
                    fps=16,
                    cfg_scale=1.0,
                    flow_shift=8.0,
                    seed=int(seed),
                    negative_prompt="",
                    sampling_algo="self_forcing",
                    num_cached_blocks=2,
                    sink_token=True,
                    num_frame_per_block=3,
                    denoising_step_list=[int(v) for v in DEFAULT_DENOISING_STEP_LIST.split(",")],
                )
                out_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
                out_dir.mkdir(parents=True, exist_ok=True)
                output_path = out_dir / "sana_wm_realtime.mp4"
                result = pipeline.generate_streaming(
                    cropped,
                    prompt_text,
                    c2w,
                    intrinsics_vec4,
                    params,
                    output_path=output_path,
                    streaming_crf=18,
                    streaming_preset="p4" if selected_encoder == "h264_nvenc" else "veryfast",
                    streaming_encoder=selected_encoder,
                    output_mode=output_mode,
                    profile_cuda=False,
                    decoded_chunk_callback=on_chunk,
                    progress_callback=on_progress,
                )
                elapsed = time.perf_counter() - started
                put_update(
                    "done",
                    {
                        "path": str(result["output_path"]) if result["output_path"] is not None else None,
                        "frames": result["n_pixel_frames"],
                        "wall": result["wall_seconds"],
                        "rt": result["realtime_factor"],
                        "elapsed": elapsed,
                        "encoder": selected_encoder if output_mode == "mp4" else "preview-only",
                        "output_mode": output_mode,
                    },
                )
            except Exception:
                put_update("error", traceback.format_exc())

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        latest = None
        video = None
        progress_state: dict[str, object] = {
            "message": "queued",
            "decode_done": 0,
            "decode_total": 0,
            "frames": 0,
        }
        progress = _progress_html(
            save_mp4=bool(save_mp4),
            encoder=selected_encoder if output_mode == "mp4" else "preview-only",
            message="queued",
        )
        status = "starting"
        yield latest, video, progress, status
        while thread.is_alive() or not updates.empty():
            try:
                kind, value = updates.get(timeout=0.5)
            except queue.Empty:
                progress = _progress_html(
                    save_mp4=bool(save_mp4),
                    encoder=selected_encoder if output_mode == "mp4" else "preview-only",
                    message=str(progress_state.get("message", status)),
                    frames=int(progress_state.get("frames", 0) or 0),
                    done=int(progress_state.get("decode_done", 0) or 0),
                    total=int(progress_state.get("decode_total", 0) or 0),
                    elapsed=time.perf_counter() - started,
                )
                yield latest, video, progress, status
                continue
            if kind == "status":
                status = str(value)
                progress_state["message"] = status
            elif kind == "frame":
                frame, detail = value
                latest = _make_preview(frame)
                status = f"{detail}  elapsed {time.perf_counter() - started:.1f}s"
            elif kind == "progress":
                progress_state.update(value)
                status = str(progress_state.get("message", status))
            elif kind == "done":
                result = value
                video = result["path"]
                progress_state.update(
                    {
                        "message": "complete",
                        "decode_done": progress_state.get("decode_total", 0),
                        "frames": result["frames"],
                    }
                )
                status = (
                    f"done: {result['frames']} frames, wall {result['wall']:.2f}s, "
                    f"{result['rt']:.3f}x realtime, encoder={result['encoder']}"
                )
                progress = _progress_html(
                    save_mp4=bool(save_mp4),
                    encoder=str(result["encoder"]),
                    message="complete",
                    frames=int(result["frames"]),
                    done=int(progress_state.get("decode_done", 0) or 0),
                    total=int(progress_state.get("decode_total", 0) or 0),
                    elapsed=float(result["elapsed"]),
                    complete=True,
                    path=result["path"],
                )
            elif kind == "error":
                status = str(value)
                progress_state["message"] = "error"
            if kind != "done":
                progress = _progress_html(
                    save_mp4=bool(save_mp4),
                    encoder=selected_encoder if output_mode == "mp4" else "preview-only",
                    message=str(progress_state.get("message", status)),
                    frames=int(progress_state.get("frames", 0) or 0),
                    done=int(progress_state.get("decode_done", 0) or 0),
                    total=int(progress_state.get("decode_total", 0) or 0),
                    elapsed=time.perf_counter() - started,
                )
            yield latest, video, progress, status

    css = """
    .gradio-container { max-width: 1180px !important; }
    #preview img { object-fit: contain; }
    .mp4-progress { border: 1px solid #333; border-radius: 6px; padding: 10px 12px; background: #161616; }
    .mp4-progress-top { display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 8px; }
    .mp4-progress-bar { height: 10px; background: #2a2a2a; border-radius: 999px; overflow: hidden; }
    .mp4-progress-bar > div { height: 100%; background: #3b82f6; transition: width 160ms linear; }
    .mp4-progress-meta, .mp4-progress-path { margin-top: 8px; color: #cfcfcf; font-size: 12px; }
    .mp4-progress-path { word-break: break-all; }
    """
    with gr.Blocks(css=css, title="Sana-WM Realtime") as demo:
        gr.Markdown("Sana-WM Realtime")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(value="asset/sana_wm/demo_0.png", type="pil", label="First frame")
                prompt = gr.Textbox(value=_default_prompt(), lines=5, label="Prompt")
                action = gr.Textbox(value=DEFAULT_ACTION, label="Action")
                intrinsics = gr.File(label="Intrinsics .npy", file_types=[".npy"])
                with gr.Row():
                    num_frames = gr.Slider(161, 961, value=961, step=8, label="Frames")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                with gr.Row():
                    translation = gr.Number(value=0.055, label="Translation")
                    rotation = gr.Number(value=1.2, label="Rotation deg")
                encoder = gr.Radio(["h264_nvenc", "libx264"], value="h264_nvenc", label="MP4 encoder")
                save_mp4 = gr.Checkbox(value=True, label="Save final MP4")
                run = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                preview = gr.Image(label="Live preview", elem_id="preview")
                video = gr.Video(label="Final MP4")
                mp4_progress = gr.HTML(
                    _progress_html(save_mp4=True, encoder="h264_nvenc", message="idle"),
                    label="Final MP4 progress",
                )
                status = gr.Textbox(label="Status", lines=4)

        run.click(
            generate,
            inputs=[
                image,
                prompt,
                action,
                num_frames,
                translation,
                rotation,
                seed,
                save_mp4,
                encoder,
                intrinsics,
            ],
            outputs=[preview, video, mp4_progress, status],
            show_progress=True,
            concurrency_limit=1,
        )
    return demo


def main() -> None:
    args = _build_parser().parse_args()
    state = DemoState(streaming_root=args.streaming_root, no_compile=args.no_compile)
    if not args.lazy_load:
        _preload_pipeline(state)
    demo = build_demo(args, state)
    demo.queue(max_size=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        debug=False,
    )


if __name__ == "__main__":
    main()
