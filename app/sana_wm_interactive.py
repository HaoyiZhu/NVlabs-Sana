#!/usr/bin/env python
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Licensed under the Apache License, Version 2.0
# SPDX-License-Identifier: Apache-2.0

"""Interactive engine for the realtime SANA-WM demo.

Drives the NVFP4-optimised chunk-pipelined streaming path in response to live
keyboard input. WASD translate the camera, the arrow keys rotate it; each
chunk's camera is injected in-place a hair before the sampler reads it, so the
autoregressive stream steers in real time. Decoded frames are JPEG-encoded on a
background thread and pushed into an asyncio queue for the browser canvas.

Adapted from the FastAPI streaming demo for the realtime pipeline, whose
``run_streaming_inference`` is callback-based (``decoded_chunk_callback``) and
whose ``sample_chunks`` yields once per chunk (no KV-save sentinel) — so the
next chunk's camera is written right *after* each yield.

Smoothing / feel:
  * Per-frame target velocity from currently-held keys.
  * Exponential low-pass on velocity (tau=0.45s pressed, tau=1.0s coasting).
  * Auto-walk: on Start the camera glides gently forward until the first key.
  * Intro precache: the first few chunks (deterministic forward glide) are
    rendered once per scene at warmup and replayed instantly on Start, so the
    cold first-chunk latency is hidden — the stream feels seamless.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pyrallis
import torch
from PIL import Image
from torchvision import transforms as T

from diffusion.model.builder import vae_encode
from diffusion.model.ltx2 import CausalVaeStreamingDecoder
from diffusion.refiner.diffusers_ltx2_refiner import (
    RefinerChunkRunner,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from diffusion.scheduler.self_forcing_flow_euler_sampler import SelfForcingFlowEulerCamCtrl
from diffusion.utils.cam_utils import compute_raymap
from diffusion.utils.chunk_utils import get_chunk_index_from_config
from diffusion.utils.logger import get_root_logger
from inference_video_scripts.inference_sana_wm import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    InferenceConfig,
    RefinerSettings,
    SanaWMPipeline,
    load_intrinsics,
    resize_and_center_crop,
    transform_intrinsics_for_crop,
)
from inference_video_scripts.streaming_pipeline import (
    StreamingPipelineConfig,
    run_streaming_inference,
)
from inference_video_scripts.camera_control import (  # shared camera-control core (demo + inference)
    DEFAULT_PITCH_LIMIT_DEG,
    DEFAULT_ROTATION_SPEED_DEG,
    DEFAULT_TRANSLATION_SPEED,
    DEMO_KEY_TO_CONTROL,
    CameraPoseIntegrator,
    VelocityState,
    controls_to_target_velocity,
)

# ============================================================================
# Constants
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "asset" / "sana_wm"
DEFAULT_STREAMING_ROOT = REPO_ROOT / "pretrained_models" / "sana_wm_streaming"

FPS = 16
NUM_FRAME_PER_BLOCK = 3
VAE_TIME_STRIDE = 8
MAX_LATENT_FRAMES = 40
MAX_PIXEL_FRAMES = (MAX_LATENT_FRAMES - 1) * VAE_TIME_STRIDE + 1  # = 313
MAX_SECONDS_ACTUAL = (MAX_PIXEL_FRAMES - 1) / FPS  # = 19.5


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


# Camera-control magnitudes come from camera_control (shared with inference);
# env vars allow per-run tuning. Smoothing (tau) lives in camera_control too.
TRANSLATION_SPEED = _env_float("SANA_WM_DEMO_TRANSLATION", DEFAULT_TRANSLATION_SPEED)
ROTATION_SPEED_RAD = math.radians(_env_float("SANA_WM_DEMO_ROTATION_DEG", DEFAULT_ROTATION_SPEED_DEG))
PITCH_LIMIT_RAD = math.radians(_env_float("SANA_WM_DEMO_PITCH_LIMIT_DEG", DEFAULT_PITCH_LIMIT_DEG))
# Gentle auto-walk speed used before the first keypress (and during the intro).
AUTO_FORWARD_SPEED = _env_float("SANA_WM_DEMO_AUTOWALK", 0.6 * TRANSLATION_SPEED)

# Streaming distilled student recipe.
DEFAULT_DENOISING_STEP_LIST = [1000, 960, 889, 727, 0]

# Number of leading decode chunks rendered once per scene and replayed
# instantly on Start (hides cold first-chunk latency). The same chunks are
# camera-locked to the forward glide in the live session, so the cached frames
# and the live frames are bit-for-bit continuous at the handoff.
INTRO_DECODED_CHUNKS = _env_int("SANA_WM_DEMO_INTRO_CHUNKS", 2)

# Browser stream pacing.
JPEG_QUALITY = _env_int("SANA_WM_DEMO_JPEG_QUALITY", 80)
SERVER_QUEUE_HIGHWATER = 64  # ~4 s @ 16 fps; drop oldest on overflow.

LOGGER = logging.getLogger("sana_wm_realtime")


# ============================================================================
# Scene registry
# ============================================================================


@dataclass(frozen=True)
class Scene:
    """One bundled demo scene. The prompt and intrinsics are baked in."""

    id: str
    label: str
    image_path: Path
    prompt_path: Path
    intr_path: Path


SCENES: list[Scene] = [
    Scene("demo_0", "Desert lakebed", ASSET_DIR / "demo_0.png",
          ASSET_DIR / "demo_0.txt", ASSET_DIR / "demo_0_intrinsics.npy"),
    Scene("demo_1", "Limestone cave", ASSET_DIR / "demo_1.png",
          ASSET_DIR / "demo_1.txt", ASSET_DIR / "demo_1_intrinsics.npy"),
    Scene("demo_2", "Mushroom forest", ASSET_DIR / "demo_2.png",
          ASSET_DIR / "demo_2.txt", ASSET_DIR / "demo_2_intrinsics.npy"),
]
SCENE_BY_ID: dict[str, Scene] = {s.id: s for s in SCENES}


# Control scheme (unified with the inference DSL via camera_control):
#   W/S = forward/back, A/D = yaw, Arrow Up/Down = pitch, Arrow Left/Right = strafe.
ALLOWED_KEYS = frozenset(DEMO_KEY_TO_CONTROL)


# ============================================================================
# Session-side state
# ============================================================================


@dataclass
class Session:
    """All state for one (single) interactive session."""

    scene: Scene
    keys_down: set[str] = field(default_factory=set)
    keys_lock: threading.Lock = field(default_factory=threading.Lock)

    velocity: VelocityState = field(default_factory=VelocityState)
    integrator: CameraPoseIntegrator = field(default_factory=lambda: CameraPoseIntegrator(PITCH_LIMIT_RAD))

    pixel_poses_full: np.ndarray | None = None  # set in _run_session_inner
    last_integrated_pixel: int = 1  # pixel 0 is identity; integration starts at 1

    # Until this pixel the camera is force-locked to the forward glide and
    # ignores keys, so the live stream reproduces the cached intro exactly.
    intro_pixel_lock: int = 0
    # True until the user presses a movement key; the camera auto-walks forward.
    auto_forward: bool = True

    stop_event: threading.Event = field(default_factory=threading.Event)
    finished_event: threading.Event = field(default_factory=threading.Event)
    frame_q: "asyncio.Queue | None" = None  # set by /ctrl
    loop: "asyncio.AbstractEventLoop | None" = None  # set by /ctrl
    n_chunks_emitted: int = 0
    max_chunks: int | None = None  # warmup / precache stop after this many chunks
    _last_keys: set[str] = field(default_factory=set)

    def snapshot_keys(self) -> set[str]:
        with self.keys_lock:
            return set(self.keys_down)

    def set_keys(self, keys: set[str]) -> None:
        with self.keys_lock:
            self.keys_down = set(keys)

    def integrate_pixels_up_to(self, target_pixel_exclusive: int) -> None:
        """Advance the velocity model + camera pose to ``target_pixel_exclusive``.

        During the intro lock the keys are ignored and the camera glides
        forward (so the live frames match the cached intro). Afterwards, before
        the first keypress, it keeps gliding forward; once any key is pressed
        ``auto_forward`` latches off and full WASD/arrow control takes over,
        with a smooth coast back to rest on release.
        """
        dt = 1.0 / FPS
        while self.last_integrated_pixel < target_pixel_exclusive:
            in_intro = self.last_integrated_pixel < self.intro_pixel_lock
            keys = set() if in_intro else self.snapshot_keys()
            if keys and self.auto_forward and not in_intro:
                self.auto_forward = False

            if self.auto_forward and not keys:
                target = VelocityState(tx=AUTO_FORWARD_SPEED)
            else:
                controls = {DEMO_KEY_TO_CONTROL[k] for k in keys if k in DEMO_KEY_TO_CONTROL}
                target = controls_to_target_velocity(
                    controls, translation_speed=TRANSLATION_SPEED, rotation_speed_rad=ROTATION_SPEED_RAD
                )

            if keys - self._last_keys:
                self.velocity.snap_to(target)  # fresh press overrides momentum
            else:
                self.velocity.step_toward(target, dt)
            self._last_keys = keys

            self.pixel_poses_full[self.last_integrated_pixel] = self.integrator.step(self.velocity)
            self.last_integrated_pixel += 1


# ============================================================================
# Camera-tensor in-place injection (matches prepare_camera / _pack_camera_conditions)
# ============================================================================


def _build_chunk_indices(total_latents: int, base_chunk: int) -> list[int]:
    """Mirror ``SelfForcingFlowEulerCamCtrl.create_autoregressive_segments``."""
    remained = total_latents % base_chunk
    n_chunks = total_latents // base_chunk
    idx = [0]
    for i in range(n_chunks):
        cur = idx[-1] + base_chunk
        if i == 0:
            cur += remained
        idx.append(cur)
    return idx


class CameraInjector:
    """Writes ``raymap`` + ``chunk_plucker`` slices for a range of latents in
    place, so the sampler picks up new motion when it reads each chunk."""

    def __init__(
        self,
        raymap_full: torch.Tensor,
        chunk_plucker_full: torch.Tensor,
        intrinsics_pixel_vec4: np.ndarray,
        pixel_h: int,
        pixel_w: int,
        latent_h: int,
        latent_w: int,
    ) -> None:
        self.raymap_full = raymap_full  # (1, T_max, 20)
        self.chunk_plucker_full = chunk_plucker_full  # (1, S*6, T_max, H, W)
        self.S = VAE_TIME_STRIDE
        self.pixel_h = pixel_h
        self.pixel_w = pixel_w
        self.latent_h = latent_h
        self.latent_w = latent_w

        intr_latent = intrinsics_pixel_vec4.astype(np.float32).copy()
        intr_latent[0] *= latent_w / float(pixel_w)
        intr_latent[2] *= latent_w / float(pixel_w)
        intr_latent[1] *= latent_h / float(pixel_h)
        intr_latent[3] *= latent_h / float(pixel_h)
        self._intr_latent = intr_latent  # (4,)

    def write_latents(self, lat_start: int, lat_end: int, pixel_poses_full: np.ndarray) -> None:
        S = self.S
        n_lat = lat_end - lat_start
        device = self.raymap_full.device
        dtype = self.raymap_full.dtype

        # Plucker pixel window per latent. Latent k>0: pixels [k*S-(S-1), k*S+1);
        # latent 0: pixels [0, S).
        pose_slab = np.empty((n_lat, S, 4, 4), dtype=np.float32)
        for li, k in enumerate(range(lat_start, lat_end)):
            s = max(0, k * S - (S - 1))
            e = s + S
            slab = pixel_poses_full[s:e]
            if slab.shape[0] < S:
                pad = S - slab.shape[0]
                slab = np.concatenate([slab, np.broadcast_to(slab[-1:], (pad, 4, 4))], axis=0)
            pose_slab[li] = slab

        pose_t = torch.from_numpy(pose_slab).to(device=device, dtype=torch.float32)
        intr_latent_t = torch.from_numpy(
            np.broadcast_to(self._intr_latent, (n_lat, S, 4)).copy()
        ).to(device=device, dtype=torch.float32)

        pose_flat = pose_t.reshape(n_lat * S, 4, 4)
        intr_flat = intr_latent_t.reshape(n_lat * S, 4)
        plucker = compute_raymap(intr_flat, pose_flat, self.latent_h, self.latent_w, use_plucker=True)
        plucker = (
            plucker.reshape(n_lat, S, self.latent_h, self.latent_w, 6)
            .permute(0, 1, 4, 2, 3)
            .reshape(n_lat, S * 6, self.latent_h, self.latent_w)
        )
        self.chunk_plucker_full[0, :, lat_start:lat_end, :, :] = plucker.permute(1, 0, 2, 3).to(dtype=dtype)

        raymap_pixel = np.arange(lat_start, lat_end) * S
        raymap_pixel = np.minimum(raymap_pixel, pixel_poses_full.shape[0] - 1)
        raymap_poses = pixel_poses_full[raymap_pixel]
        raymap_poses_t = torch.from_numpy(raymap_poses.astype(np.float32)).to(device=device, dtype=torch.float32)
        intr_lat_rows = torch.from_numpy(
            np.broadcast_to(self._intr_latent, (n_lat, 4)).copy()
        ).to(device=device, dtype=torch.float32)
        raymap_rows = torch.cat([raymap_poses_t.reshape(n_lat, 16), intr_lat_rows], dim=-1).to(dtype=dtype)
        self.raymap_full[0, lat_start:lat_end, :] = raymap_rows


# ============================================================================
# Frame emitters (decoded_chunk_callback compatible: (pixel_np, frame_base, chunk_idx))
# ============================================================================


def _push_frame(q: "asyncio.Queue", jpeg: bytes) -> None:
    """Push a frame, dropping the oldest if we'd exceed the high-water mark."""
    if q.qsize() >= SERVER_QUEUE_HIGHWATER:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        q.put_nowait(jpeg)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
            q.put_nowait(jpeg)
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            pass


def _encode_jpeg(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


class LiveFrameEmitter:
    """JPEG-encodes decoded frames on a background thread and pushes them to the
    session's asyncio queue. Skips the first ``skip_pixels`` frames (already
    shown to the client from the cached intro)."""

    def __init__(self, session: Session, skip_pixels: int = 0) -> None:
        self.session = session
        self.skip = int(skip_pixels)
        self._emitted = 0
        self._closed = False
        self._pixel_q: queue.Queue = queue.Queue(maxsize=8)
        self._thread = threading.Thread(target=self._loop, name="sana-wm-jpeg", daemon=True)
        self._thread.start()

    def __call__(self, pixel_np: np.ndarray, frame_base: int, chunk_idx: int) -> None:
        if self._closed or pixel_np.shape[0] <= 0:
            return
        try:
            self._pixel_q.put_nowait(pixel_np)
        except queue.Full:
            pass  # encoder is far behind; drop rather than block the orchestrator

    def _loop(self) -> None:
        while True:
            try:
                item = self._pixel_q.get(timeout=0.5)
            except queue.Empty:
                if self._closed:
                    return
                continue
            if item is None:
                return
            for i in range(item.shape[0]):
                if self._emitted < self.skip:
                    self._emitted += 1
                    continue
                self._emitted += 1
                jpeg = _encode_jpeg(item[i])
                loop = self.session.loop
                q = self.session.frame_q
                if loop is not None and q is not None:
                    loop.call_soon_threadsafe(_push_frame, q, jpeg)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._pixel_q.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=2.0)


class CaptureFrameEmitter:
    """Collects the first ``max_chunks`` decode chunks' frames as JPEGs, then
    stops the session. Used by the per-scene intro precache."""

    def __init__(self, session: Session, max_chunks: int) -> None:
        self.session = session
        self.max_chunks = int(max_chunks)
        self.jpegs: list[bytes] = []
        self._chunks = 0

    def __call__(self, pixel_np: np.ndarray, frame_base: int, chunk_idx: int) -> None:
        if self._chunks >= self.max_chunks or pixel_np.shape[0] <= 0:
            return
        for i in range(pixel_np.shape[0]):
            self.jpegs.append(_encode_jpeg(pixel_np[i]))
        self._chunks += 1
        if self._chunks >= self.max_chunks:
            self.session.stop_event.set()

    def close(self) -> None:
        return


# ============================================================================
# Pipeline + per-scene cache (built once at startup)
# ============================================================================


@dataclass
class SceneCache:
    cond: torch.Tensor
    cond_mask: torch.Tensor
    neg: torch.Tensor
    neg_mask: torch.Tensor
    first_latent: torch.Tensor  # (1, C, 1, h, w)
    intrinsics_pixel_vec4: np.ndarray  # (4,)
    refiner_prompt_embeds: torch.Tensor
    refiner_prompt_attention_mask: torch.Tensor


@dataclass
class LoadedPipeline:
    pipeline: SanaWMPipeline
    scene_caches: dict[str, SceneCache]
    # Per-scene cached intro: list of JPEG frames replayed instantly on Start.
    intro_jpegs: dict[str, list[bytes]] = field(default_factory=dict)
    intro_pixel_lock: int = 0


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _resolve_paths(streaming_root: Path) -> tuple[Path, Path, Path, Path, Path]:
    cfg = streaming_root / "sana_wm_streaming_1600m_720p.yaml"
    model = streaming_root / "sana_dit" / "model.pt"
    causal_vae = streaming_root / "ltx2_causal_vae"
    refiner = streaming_root / "refiner_diffusers"
    gemma = streaming_root / "gemma3_12b"
    for label, p in [("config", cfg), ("model", model), ("causal_vae", causal_vae),
                     ("refiner", refiner), ("gemma", gemma)]:
        if not p.exists():
            raise FileNotFoundError(f"streaming {label} not found at {p}")
    return cfg, model, causal_vae, refiner, gemma


def _apply_fast_defaults() -> None:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(True)
    import torch._inductor.config as _ic
    _ic.coordinate_descent_tuning = True
    _ic.epilogue_fusion = True


# ---- Persistent disk cache for per-scene tensors --------------------------

_CACHE_SCHEMA_VERSION = 2
_DEFAULT_CACHE_DIRNAME = "sana_wm_realtime_demo"


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _resolve_cache_root(override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("SANA_WM_STREAMING_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return (Path(xdg).expanduser() / _DEFAULT_CACHE_DIRNAME).resolve()
    return (Path.home() / ".cache" / _DEFAULT_CACHE_DIRNAME).resolve()


def _config_signature(streaming_root: Path) -> str:
    parts: list[str] = []
    targets = [
        streaming_root / "sana_dit" / "model.pt",
        streaming_root / "ltx2_causal_vae",
        streaming_root / "refiner_diffusers",
        streaming_root / "gemma3_12b",
        streaming_root / "sana_wm_streaming_1600m_720p.yaml",
    ]
    for p in targets:
        if p.is_file():
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
        elif p.is_dir():
            parts.append(f"{p.name}:dir:{int(p.stat().st_mtime)}")
        else:
            parts.append(f"{p.name}:missing")
    return _hash_str("|".join(parts))


def _persistent_cache_dir(streaming_root: Path, override: str | None = None) -> Path:
    return _resolve_cache_root(override) / f"v{_CACHE_SCHEMA_VERSION}" / _config_signature(streaming_root)


def _safe_torch_save(obj: dict, path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)
        return True
    except OSError as exc:
        LOGGER.warning(f"[cache] write failed at {path}: {exc!r}")
        return False


def _try_load(path: Path, device: torch.device | str) -> dict | None:
    if not path.is_file():
        return None
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(f"[cache] load failed at {path}: {exc!r} — recomputing")
        return None


def load_pipeline(
    streaming_root: Path,
    device: torch.device,
    *,
    cache_dir: Path | None = None,
    no_compile: bool = False,
) -> LoadedPipeline:
    """Build the SanaWMPipeline, compile, NVFP4-prepare, and pre-encode scenes."""
    _apply_fast_defaults()
    cfg, model, causal_vae, refiner_root, gemma_root = _resolve_paths(streaming_root)
    config: InferenceConfig = pyrallis.parse(config_class=InferenceConfig, config_path=str(cfg), args=[])
    config.vae.vae_type = "LTX2VAE_diffusers_causal"
    config.vae.vae_pretrained = str(causal_vae)

    refiner = RefinerSettings(
        root=str(refiner_root),
        gemma_root=str(gemma_root),
        sink_size=1,
        seed=42,
        block_size=NUM_FRAME_PER_BLOCK,
        # Match the inference CLI (11) for consistent refiner temporal context.
        kv_max_frames=_env_int("SANA_WM_DEMO_REFINER_KV_MAX_FRAMES", 11),
    )

    pipeline = SanaWMPipeline(
        config=config,
        model_path=str(model),
        device=device,
        refiner=refiner,
        offload_vae=False,
        offload_refiner=False,
        offload_text_encoder=True,
        logger=LOGGER,
    )

    if not no_compile:
        compile_mode = os.environ.get("SANA_WM_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs")
        dynamic = os.environ.get("SANA_WM_TORCH_COMPILE_DYNAMIC", "0").lower() not in {"0", "false", "no", "off", ""}
        targets = {t.strip().lower() for t in os.environ.get("SANA_WM_TORCH_COMPILE_TARGETS", "refiner").split(",") if t.strip()}
        # The causal VAE streaming decoder must NOT be compiled: the compiled
        # graph corrupts its cross-chunk cache (chunk 0 fine, chunk >=1 -> blank
        # gray). Refuse it; compile the refiner (the heavy module) only.
        if "vae" in targets:
            LOGGER.warning("[load] ignoring 'vae' compile target: it corrupts the VAE streaming cache (chunk>=1 blank).")
            targets.discard("vae")
        if "refiner" in targets:
            LOGGER.info("[load] compiling refiner.transformer (%s)", compile_mode)
            pipeline.refiner.transformer = torch.compile(pipeline.refiner.transformer, mode=compile_mode, dynamic=dynamic)

    scene_caches = _build_scene_caches(pipeline, cache_dir=cache_dir)
    _prepare_resident_modules(pipeline)

    intro_lock = _intro_pixel_lock()
    loaded = LoadedPipeline(pipeline=pipeline, scene_caches=scene_caches, intro_pixel_lock=intro_lock)
    return loaded


def _prepare_resident_modules(pipeline: SanaWMPipeline) -> None:
    """Stage the NVFP4 / streaming modules onto the GPU once, resident for all
    interactive sessions (mirrors the realtime generate_streaming prep)."""
    LOGGER.info("[load] preparing resident NVFP4 / streaming modules")
    with torch.inference_mode():
        cpu_stage = os.environ.get("SANA_WM_TE_NVFP4_CPU_STAGING", "1").lower() in {"1", "true", "yes", "on"}
        if cpu_stage:
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
        # Keep the VAE decoder resident on GPU for low-latency interactive decode.
        pipeline._move_vae_decoder_for_streaming(pipeline.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _build_scene_caches(pipeline: SanaWMPipeline, cache_dir: Path | None = None) -> dict[str, SceneCache]:
    """Pre-encode prompt embeds (stage-1 + refiner) and first-frame VAE latents
    for every scene. Two-pass: load from disk; only compute (and load Gemma3 for
    the refiner) on a miss. Keyed by prompt + image hashes."""
    if cache_dir is not None:
        LOGGER.info(f"[scenes] persistent cache dir: {cache_dir}")
    device = pipeline.device
    refiner = pipeline.refiner

    scratch: dict[str, dict] = {}
    stage1_misses: list[Scene] = []
    refiner_misses: list[Scene] = []
    neg_loaded: tuple[torch.Tensor, torch.Tensor] | None = None

    neg_path = (cache_dir / "neg.pt") if cache_dir is not None else None
    if neg_path is not None:
        d = _try_load(neg_path, device)
        if d is not None and d.get("schema") == _CACHE_SCHEMA_VERSION:
            neg_loaded = (
                d["neg"].to(device=device, dtype=pipeline.weight_dtype).contiguous(),
                d["neg_mask"].to(device=device).contiguous(),
            )

    for scene in SCENES:
        prompt = _load_prompt(scene.prompt_path)
        prompt_hash = _hash_str(prompt)
        image_hash = _hash_file(scene.image_path)

        with Image.open(scene.image_path) as im:
            cropped, src_size, rsz_size, crop_off = resize_and_center_crop(im.convert("RGB"))
        intr_src = load_intrinsics(scene.intr_path, 1)
        intr_vec4 = transform_intrinsics_for_crop(intr_src, src_size, rsz_size, crop_off)[0]

        sc_scratch: dict = {
            "prompt": prompt,
            "prompt_hash": prompt_hash,
            "image_hash": image_hash,
            "cropped": cropped,
            "intrinsics_pixel_vec4": intr_vec4.astype(np.float32).copy(),
        }

        s1_path = (cache_dir / f"stage1_{scene.id}.pt") if cache_dir is not None else None
        loaded = _try_load(s1_path, device) if s1_path is not None else None
        if (loaded is not None and loaded.get("schema") == _CACHE_SCHEMA_VERSION
                and loaded.get("prompt_hash") == prompt_hash and loaded.get("image_hash") == image_hash):
            sc_scratch["cond"] = loaded["cond"].to(device).contiguous()
            sc_scratch["cond_mask"] = loaded["cond_mask"].to(device).contiguous()
            sc_scratch["first_latent"] = loaded["first_latent"].to(device).contiguous()
        else:
            stage1_misses.append(scene)

        r_path = (cache_dir / f"refiner_{scene.id}.pt") if cache_dir is not None else None
        loaded = _try_load(r_path, refiner.device) if r_path is not None else None
        if (loaded is not None and loaded.get("schema") == _CACHE_SCHEMA_VERSION
                and loaded.get("prompt_hash") == prompt_hash):
            sc_scratch["refiner_embeds"] = loaded["embeds"].to(device=refiner.device, dtype=refiner.dtype).contiguous()
            sc_scratch["refiner_mask"] = loaded["mask"].to(refiner.device).contiguous()
        else:
            refiner_misses.append(scene)

        scratch[scene.id] = sc_scratch

    LOGGER.info(
        f"[scenes] disk hits: stage1={len(SCENES) - len(stage1_misses)}/{len(SCENES)}, "
        f"refiner={len(SCENES) - len(refiner_misses)}/{len(SCENES)}, "
        f"neg={'hit' if neg_loaded is not None else 'miss'}"
    )

    refiner.transformer.to(refiner.device)
    refiner.connectors.to(refiner.device)

    # no_grad (not inference_mode): the cached tensors are later consumed by the
    # refiner's connector projection in RefinerChunkRunner.__init__, which runs
    # outside any inference-mode region — inference tensors would break autograd.
    with torch.no_grad():
        if stage1_misses or neg_loaded is None:
            for scene in stage1_misses or SCENES[:1]:
                prompt = scratch[scene.id]["prompt"]
                cond, cond_mask, neg_t, neg_mask_t = pipeline._encode_prompts(prompt, "")
                if neg_loaded is None:
                    neg_loaded = (neg_t.contiguous(), neg_mask_t.contiguous())
                    if neg_path is not None:
                        _safe_torch_save(
                            {"schema": _CACHE_SCHEMA_VERSION,
                             "neg": neg_t.detach().cpu(), "neg_mask": neg_mask_t.detach().cpu()},
                            neg_path,
                        )
                if scene in stage1_misses:
                    cropped = scratch[scene.id]["cropped"]
                    img = (T.ToTensor()(cropped) * 2.0 - 1.0).unsqueeze(0).unsqueeze(2)
                    first_latent = vae_encode(
                        pipeline.config.vae.vae_type, pipeline.vae,
                        img.to(device, dtype=pipeline.vae_dtype), device=device,
                    ).to(pipeline.weight_dtype)
                    scratch[scene.id]["cond"] = cond.contiguous()
                    scratch[scene.id]["cond_mask"] = cond_mask.contiguous()
                    scratch[scene.id]["first_latent"] = first_latent.contiguous()
                    s1_path = (cache_dir / f"stage1_{scene.id}.pt") if cache_dir is not None else None
                    if s1_path is not None:
                        _safe_torch_save(
                            {"schema": _CACHE_SCHEMA_VERSION,
                             "prompt_hash": scratch[scene.id]["prompt_hash"],
                             "image_hash": scratch[scene.id]["image_hash"],
                             "cond": cond.detach().cpu(), "cond_mask": cond_mask.detach().cpu(),
                             "first_latent": first_latent.detach().cpu()},
                            s1_path,
                        )

        if refiner_misses:
            LOGGER.info(f"[scenes] refiner cache miss for {[s.id for s in refiner_misses]} — loading Gemma3 once…")
            from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
            from diffusion.refiner.diffusers_ltx2_refiner import _pack_text_embeds

            tok = AutoTokenizer.from_pretrained(refiner.gemma_root)
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            text_encoder = (
                Gemma3ForConditionalGeneration.from_pretrained(
                    refiner.gemma_root, torch_dtype=refiner.dtype, low_cpu_mem_usage=True
                ).eval().to(refiner.device)
            )
            refiner.connectors.to(refiner.device)
            refiner.transformer.to(refiner.device)
            text_backbone = getattr(text_encoder, "model", text_encoder)

            for scene in refiner_misses:
                t0 = time.time()
                prompt = scratch[scene.id]["prompt"]
                text_inputs = tok(
                    [prompt], padding="max_length", max_length=refiner.text_max_sequence_length,
                    truncation=True, add_special_tokens=True, return_tensors="pt",
                )
                input_ids = text_inputs.input_ids.to(refiner.device)
                attn = text_inputs.attention_mask.to(refiner.device)
                outputs = text_backbone(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
                hidden = torch.stack(outputs.hidden_states, dim=-1)
                seqlens = attn.sum(dim=-1)
                prompt_embeds_packed = _pack_text_embeds(
                    hidden, seqlens, device=refiner.device, padding_side=tok.padding_side
                ).to(dtype=refiner.dtype)
                conn_embeds, _, conn_mask = refiner.connectors(prompt_embeds_packed, attn)
                scratch[scene.id]["refiner_embeds"] = conn_embeds.to(device=refiner.device, dtype=refiner.dtype).contiguous()
                scratch[scene.id]["refiner_mask"] = conn_mask.to(refiner.device).contiguous()
                r_path = (cache_dir / f"refiner_{scene.id}.pt") if cache_dir is not None else None
                if r_path is not None:
                    _safe_torch_save(
                        {"schema": _CACHE_SCHEMA_VERSION, "prompt_hash": scratch[scene.id]["prompt_hash"],
                         "embeds": conn_embeds.detach().cpu(), "mask": conn_mask.detach().cpu()},
                        r_path,
                    )
                LOGGER.info(f"[scenes] refiner-encoded {scene.id!r} in {time.time() - t0:.1f}s")

            del text_encoder, text_backbone
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            LOGGER.info("[scenes] refiner cache fully hit — Gemma3 NOT loaded")

    if neg_loaded is None:
        raise RuntimeError("Failed to compute or load the negative embedding")

    out: dict[str, SceneCache] = {}
    for scene in SCENES:
        sc = scratch[scene.id]
        out[scene.id] = SceneCache(
            cond=sc["cond"],
            cond_mask=sc["cond_mask"],
            neg=neg_loaded[0].to(device=device, dtype=pipeline.weight_dtype).contiguous(),
            neg_mask=neg_loaded[1].to(device).contiguous(),
            first_latent=sc["first_latent"],
            intrinsics_pixel_vec4=sc["intrinsics_pixel_vec4"],
            refiner_prompt_embeds=sc["refiner_embeds"],
            refiner_prompt_attention_mask=sc["refiner_mask"],
        )
    return out


def _intro_pixel_lock() -> int:
    """Pixel index up to which the live camera is forced to the forward glide so
    it reproduces the cached intro. Covers the first INTRO_DECODED_CHUNKS chunks."""
    if INTRO_DECODED_CHUNKS <= 0:
        return 0
    chunk_indices = _build_chunk_indices(MAX_LATENT_FRAMES, NUM_FRAME_PER_BLOCK)
    n_chunks = len(chunk_indices) - 1
    k = min(INTRO_DECODED_CHUNKS, n_chunks)
    return min(MAX_PIXEL_FRAMES, (chunk_indices[k] - 1) * VAE_TIME_STRIDE + VAE_TIME_STRIDE)


# ============================================================================
# Session loop (producer thread)
# ============================================================================


@torch.no_grad()
def _run_session_inner(
    loaded: LoadedPipeline,
    session: Session,
    *,
    frame_callback: Callable[[np.ndarray, int, int], None] | None,
    output_mode: str,
    skip_pixels: int = 0,
) -> None:
    pipeline = loaded.pipeline
    scene = session.scene
    sc = loaded.scene_caches[scene.id]

    config = pipeline.config
    device = pipeline.device
    vae_stride = config.vae.vae_stride
    latent_T = MAX_LATENT_FRAMES
    latent_h = TARGET_HEIGHT // vae_stride[-1]
    latent_w = TARGET_WIDTH // vae_stride[-1]
    n_pixel_total = MAX_PIXEL_FRAMES
    S = VAE_TIME_STRIDE

    raymap_dim = 16 + 4
    raymap_full = torch.empty(1, latent_T, raymap_dim, device=device, dtype=pipeline.weight_dtype)
    chunk_plucker_full = torch.empty(
        1, S * 6, latent_T, latent_h, latent_w, device=device, dtype=pipeline.weight_dtype
    )

    pixel_poses_full = np.tile(np.eye(4, dtype=np.float64)[None], (n_pixel_total, 1, 1))
    session.pixel_poses_full = pixel_poses_full
    session.last_integrated_pixel = 1
    session.intro_pixel_lock = loaded.intro_pixel_lock

    injector = CameraInjector(
        raymap_full=raymap_full, chunk_plucker_full=chunk_plucker_full,
        intrinsics_pixel_vec4=sc.intrinsics_pixel_vec4,
        pixel_h=TARGET_HEIGHT, pixel_w=TARGET_WIDTH, latent_h=latent_h, latent_w=latent_w,
    )

    cond, neg, cond_mask = sc.cond, sc.neg, sc.cond_mask
    chunk_index_extra = get_chunk_index_from_config(config, num_frames=latent_T)
    model_kwargs: dict = dict(
        data_info={
            "img_hw": torch.tensor([[TARGET_HEIGHT, TARGET_WIDTH]], dtype=torch.float, device=device),
            "condition_frame_info": {0: 0.0},
        },
        mask=cond_mask,
        camera_conditions=raymap_full,
        chunk_plucker=chunk_plucker_full,
    )
    if chunk_index_extra is not None:
        model_kwargs["chunk_index"] = chunk_index_extra

    flow_shift = pipeline._resolve_flow_shift(8.0)
    solver = SelfForcingFlowEulerCamCtrl(
        pipeline.model,
        condition=cond,
        uncondition=neg,
        cfg_scale=1.0,
        flow_shift=flow_shift,
        model_kwargs=model_kwargs,
        base_chunk_frames=NUM_FRAME_PER_BLOCK,
        num_cached_blocks=2,
        sink_token=True,
        use_softmax_attention=True,
    )
    chunk_indices = [int(v) for v in solver.create_autoregressive_segments(latent_T)]
    n_chunks = len(chunk_indices) - 1

    # Seed chunk 0's camera with the forward glide before the first sampler read.
    target_pixel_0 = min(n_pixel_total, (chunk_indices[1] - 1) * S + S)
    session.integrate_pixels_up_to(target_pixel_0)
    injector.write_latents(0, chunk_indices[1], pixel_poses_full)

    generator = torch.Generator(device=device).manual_seed(42)
    z = torch.randn(
        1, sc.first_latent.shape[1], latent_T, latent_h, latent_w,
        dtype=pipeline.weight_dtype, device=device, generator=generator,
    )
    z[:, :, :1] = sc.first_latent

    inner_iter = solver.sample_chunks(
        z, generator=generator, denoising_step_list=list(DEFAULT_DENOISING_STEP_LIST)
    )

    def injected_iter() -> Iterator[tuple[int, torch.Tensor, int, int]]:
        for item in inner_iter:
            if session.stop_event.is_set():
                return
            k, latent_view, sf, ef = item
            session.n_chunks_emitted = k + 1
            yield (k, latent_view, sf, ef)
            # Resumed when the orchestrator wants the next chunk: integrate live
            # input and write chunk k+1's camera before the sampler reads it.
            next_k = k + 1
            if next_k < n_chunks:
                next_sf = chunk_indices[next_k]
                next_ef = chunk_indices[next_k + 1]
                target_pixel_excl = min(n_pixel_total, (next_ef - 1) * S + S)
                session.integrate_pixels_up_to(target_pixel_excl)
                injector.write_latents(next_sf, next_ef, pixel_poses_full)
            if session.max_chunks is not None and session.n_chunks_emitted >= session.max_chunks:
                session.stop_event.set()

    sigmas_t = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device)
    refiner_runner = RefinerChunkRunner(
        pipeline.refiner,
        prompt_embeds=sc.refiner_prompt_embeds,
        prompt_attention_mask=sc.refiner_prompt_attention_mask,
        fps=float(FPS),
        sigmas=sigmas_t,
        source_sink_frames=int(pipeline.refiner_settings.sink_size),
        block_size=int(pipeline.refiner_settings.block_size),
        kv_max_frames=int(pipeline.refiner_settings.kv_max_frames),
        seed=int(pipeline.refiner_settings.seed),
        spatial_shape=(int(z.shape[3]), int(z.shape[4])),
        n_active_frames=max(int(z.shape[2]) - int(pipeline.refiner_settings.sink_size), 0),
        latent_channels=int(z.shape[1]),
        batch_size=int(z.shape[0]),
    )
    vae_streaming_decoder = CausalVaeStreamingDecoder(pipeline.vae)

    cfg = StreamingPipelineConfig(
        sink_size=int(pipeline.refiner_settings.sink_size),
        block_size=int(pipeline.refiner_settings.block_size),
        fps=FPS,
        output_path="/tmp/sana_wm_realtime_unused.mp4",
        drop_first_pixel=True,
        output_mode=output_mode,
        lazy_vae_decoder=False,
        stage1_chunk_ends=tuple(chunk_indices[1:]),
        decoded_chunk_callback=frame_callback,
    )

    LOGGER.info(f"[session] starting {scene.id} (latent_T={latent_T}, n_chunks={n_chunks}, skip={skip_pixels})")
    t0 = time.time()
    try:
        run_streaming_inference(
            stage1_chunk_iter=injected_iter(),
            n_stage1_chunks=n_chunks,
            z_init=z,
            refiner_runner=refiner_runner,
            vae_streaming_decoder=vae_streaming_decoder,
            pixel_h=TARGET_HEIGHT,
            pixel_w=TARGET_WIDTH,
            config=cfg,
            logger=LOGGER,
        )
        LOGGER.info(f"[session] {scene.id} done in {time.time() - t0:.1f}s")
    except StopIteration:
        # injected_iter returned early on stop_event (Reset / intro precache).
        # The orchestrator pulls stage-1 via next(); an early stop surfaces as
        # StopIteration. Treat as a clean stop; re-raise only if unexpected.
        if not session.stop_event.is_set():
            raise
        LOGGER.info(f"[session] {scene.id} stopped early at chunk {session.n_chunks_emitted} ({time.time() - t0:.1f}s)")
    finally:
        # Drain in-flight refiner/decode stream work before the GPU is reused by
        # the next session, so sessions never race kernels.
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def run_session(loaded: LoadedPipeline, session: Session) -> None:
    """Drive one live interactive session on a background thread."""
    emitter = LiveFrameEmitter(session, skip_pixels=len(loaded.intro_jpegs.get(session.scene.id, [])))
    try:
        _run_session_inner(loaded, session, frame_callback=emitter, output_mode="cpu")
    except Exception:
        LOGGER.exception("[session] producer thread crashed")
    finally:
        emitter.close()
        session.finished_event.set()


def warmup(loaded: LoadedPipeline, passes: int = 2) -> None:
    """Prime torch.compile with full dummy sessions (held at the forward glide)."""
    LOGGER.info(f"[warmup] running {passes} full session(s) to prime torch.compile…")
    t_total = time.time()
    for i in range(passes):
        t0 = time.time()
        s = Session(scene=SCENES[0])
        try:
            _run_session_inner(loaded, s, frame_callback=None, output_mode="discard")
        except Exception:
            LOGGER.exception(f"[warmup] pass {i + 1}/{passes} failed")
            break
        finally:
            s.finished_event.set()
        LOGGER.info(f"[warmup] pass {i + 1}/{passes} done in {time.time() - t0:.1f}s")
    LOGGER.info(f"[warmup] total {time.time() - t_total:.1f}s")


def precache_intro(loaded: LoadedPipeline) -> None:
    """Render the first INTRO_DECODED_CHUNKS decode chunks per scene (forward
    glide) and stash the JPEGs for instant replay on Start."""
    if INTRO_DECODED_CHUNKS <= 0:
        LOGGER.info("[intro] precache disabled (INTRO_DECODED_CHUNKS<=0)")
        return
    for scene in SCENES:
        t0 = time.time()
        s = Session(scene=scene)
        capture = CaptureFrameEmitter(s, max_chunks=INTRO_DECODED_CHUNKS)
        try:
            _run_session_inner(loaded, s, frame_callback=capture, output_mode="cpu")
        except Exception:
            LOGGER.exception(f"[intro] precache for {scene.id} failed")
        finally:
            s.finished_event.set()
        loaded.intro_jpegs[scene.id] = capture.jpegs
        LOGGER.info(
            f"[intro] cached {len(capture.jpegs)} frames for {scene.id} in {time.time() - t0:.1f}s"
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_runtime(loaded: LoadedPipeline, *, do_warmup: bool = True) -> None:
    """Warm compile + precompute per-scene intro frames."""
    if do_warmup:
        warmup(loaded)
    else:
        LOGGER.warning("[startup] warmup skipped — first session pays the cold compile cost")
    precache_intro(loaded)
    LOGGER.info("[startup] runtime ready")
