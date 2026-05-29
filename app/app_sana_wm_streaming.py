#!/usr/bin/env python
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Licensed under the Apache License, Version 2.0
# SPDX-License-Identifier: Apache-2.0

"""Interactive streaming SANA-WM demo.

A FastAPI server that drives the chunk-pipelined streaming pipeline in
response to live WASD/IJKL keyboard input from a single browser session.
Decoded frames flow into a queue and out to the browser one frame at a time
over a WebSocket, drawn onto a ``<canvas>`` at 16 fps. The user picks one of
the bundled scenes (desert / cave / mushroom), presses Start, and steers for
up to 20 s.

Smoothing:
  * Per-frame target velocity from currently-held keys.
  * Exponential low-pass on velocity (tau=0.45s pressed, tau=1.0s coasting).
  * Inertial coast on key release — no snap to zero.
"""

from __future__ import annotations

import os

# Env knobs that must be set before any torch / diffusion.* import. Mirrors
# inference_video_scripts/inference_sana_wm_streaming.py exactly so the
# compiled artifacts cache hits.
os.environ.setdefault("DISABLE_XFORMERS", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Iterator  # noqa: E402

import numpy as np  # noqa: E402
import pyrallis  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from diffusion.model.builder import vae_encode  # noqa: E402
from diffusion.model.ltx2 import CausalVaeStreamingDecoder  # noqa: E402
from diffusion.refiner.diffusers_ltx2_refiner import (  # noqa: E402
    RefinerChunkRunner,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from diffusion.scheduler.self_forcing_flow_euler_sampler import (  # noqa: E402
    SelfForcingFlowEulerCamCtrl,
)
from diffusion.utils.cam_utils import compute_raymap  # noqa: E402
from diffusion.utils.logger import get_root_logger  # noqa: E402
from diffusion.utils.chunk_utils import get_chunk_index_from_config  # noqa: E402
from inference_video_scripts.inference_sana_wm import (  # noqa: E402
    TARGET_HEIGHT,
    TARGET_WIDTH,
    InferenceConfig,
    RefinerSettings,
    SanaWMPipeline,
    load_intrinsics,
    resize_and_center_crop,
    transform_intrinsics_for_crop,
)
from inference_video_scripts.streaming_pipeline import (  # noqa: E402
    StreamingPipelineConfig,
    run_streaming_inference,
)

# ============================================================================
# Constants
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "asset" / "sana_wm"
DEFAULT_STREAMING_ROOT = REPO_ROOT / "pretrained_models" / "sana_wm_streaming"

# Hard duration cap. Two constraints from the streaming pipeline:
#   (a) LTX-2 VAE requires pixel_frames = 8k+1.
#   (b) ``streaming_pipeline.run_streaming_inference`` requires
#       ``(latent_T - sink_size) % block_size == 0``.
# With sink_size=1, block_size=3: latent_T must satisfy (latent_T - 1) % 3 == 0,
# i.e. k must be divisible by 3. So the legal frame counts at 16 fps are
# 8*k+1 = 1, 25, 49, 73, 97, ..., 313, 337, ...; the largest at-or-under 20 s
# is **313 pixel frames = 19.5 s** (latent_T = 40, 13 stage-1 chunks).
FPS = 16
NUM_FRAME_PER_BLOCK = 3
VAE_TIME_STRIDE = 8
PIXEL_FRAMES_PER_LATENT = VAE_TIME_STRIDE
MAX_LATENT_FRAMES = 40
MAX_PIXEL_FRAMES = (MAX_LATENT_FRAMES - 1) * VAE_TIME_STRIDE + 1  # = 313
MAX_SECONDS_ACTUAL = (MAX_PIXEL_FRAMES - 1) / FPS  # = 19.5

# Velocity model.
PEAK_TRANSLATION_PER_FRAME = 0.025  # half the bidirectional default
PEAK_ROTATION_RAD_PER_FRAME = math.radians(0.6)  # half the bidirectional default
PITCH_LIMIT_RAD = math.radians(60.0)
TAU_PRESS = 0.45  # seconds — exponential time constant when keys are pressed
TAU_COAST = 1.0  # seconds — slower decay when all keys released

# Streaming distilled student recipe.
DEFAULT_DENOISING_STEP_LIST = [1000, 960, 889, 727, 0]
DEFAULT_REFINER_SIGMAS = STAGE_2_DISTILLED_SIGMA_VALUES

# Browser stream pacing.
JPEG_QUALITY = 78
SERVER_QUEUE_HIGHWATER = 64  # ~4 s @ 16 fps; drop oldest on overflow.

# Network defaults.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7860

LOGGER = logging.getLogger("sana_wm_streaming")


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


# ============================================================================
# Pose math (mirrors action_string_to_c2w)
# ============================================================================


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


@dataclass
class VelocityState:
    """Per-frame velocity in the same units as the bidirectional action DSL."""

    tx: float = 0.0  # forward (+) / back (-), per-frame translation magnitude
    sx: float = 0.0  # strafe right (+) / left (-)
    yaw: float = 0.0  # rad/frame
    pitch: float = 0.0  # rad/frame

    def step_toward(self, target: "VelocityState", dt: float) -> None:
        """Exponential low-pass on each axis. Coast tau is used when the
        target is zero (key released) so motion decays gently instead of
        snapping to a halt."""
        for attr in ("tx", "sx", "yaw", "pitch"):
            cur = getattr(self, attr)
            tgt = getattr(target, attr)
            tau = TAU_PRESS if abs(tgt) > 1e-12 else TAU_COAST
            alpha = 1.0 - math.exp(-dt / tau)
            setattr(self, attr, cur + alpha * (tgt - cur))


def _target_velocity_from_keys(keys: set[str]) -> VelocityState:
    """Map the live key set to a target per-frame velocity vector.

    Driving-style controls: A/D *steer* (yaw) rather than strafe.

      * W / S   — forward / back translation along current heading.
      * A / J   — yaw left   (equivalent; ``a`` is the WASD ergonomic, ``j``
                  matches the original DSL).
      * D / L   — yaw right  (likewise).
      * I / K   — pitch up / down.

    Combined with W, A/D produce a smooth forward arc (W+D = walk forward
    while curving right), which is what users expect from a game-style
    camera. Pure lateral strafe is gone — the trajectory math is identical
    to pressing J/L while holding W.
    """
    fwd = (1.0 if "w" in keys else 0.0) - (1.0 if "s" in keys else 0.0)
    yaw_left = ("a" in keys) or ("j" in keys)
    yaw_right = ("d" in keys) or ("l" in keys)
    yaw = (1.0 if yaw_right else 0.0) - (1.0 if yaw_left else 0.0)
    pit = (1.0 if "i" in keys else 0.0) - (1.0 if "k" in keys else 0.0)
    return VelocityState(
        tx=fwd * PEAK_TRANSLATION_PER_FRAME,
        sx=0.0,
        yaw=yaw * PEAK_ROTATION_RAD_PER_FRAME,
        pitch=pit * PEAK_ROTATION_RAD_PER_FRAME,
    )


# ============================================================================
# Session-side state
# ============================================================================


@dataclass
class Session:
    """All state for one (single) interactive session.

    The producer thread reads/writes ``keys_down`` (under a lock); the FastAPI
    /ctrl handler writes ``keys_down`` on every keydown/keyup transition.
    """

    scene: Scene
    keys_down: set[str] = field(default_factory=set)
    keys_lock: threading.Lock = field(default_factory=threading.Lock)

    velocity: VelocityState = field(default_factory=VelocityState)
    current_pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    current_pitch: float = 0.0

    pixel_poses_full: np.ndarray = field(default=None)  # set in factory
    last_integrated_pixel: int = 1  # pixel 0 is identity; integration starts at 1

    stop_event: threading.Event = field(default_factory=threading.Event)
    finished_event: threading.Event = field(default_factory=threading.Event)
    # ``armed_event`` fires once the producer has built the solver / runner /
    # decoder and is parked waiting for the user. ``fire_event`` releases it
    # to begin generation — set when the client presses a steering key (or
    # sends an explicit ``fire`` op). The warmup pre-sets ``fire_event`` so
    # it never blocks.
    armed_event: threading.Event = field(default_factory=threading.Event)
    fire_event: threading.Event = field(default_factory=threading.Event)
    frame_q: asyncio.Queue | None = None  # set by /ctrl
    loop: asyncio.AbstractEventLoop | None = None  # set by /ctrl
    n_chunks_emitted: int = 0
    # If set, the producer stops cleanly after this many chunks. Used by the
    # startup warmup to prime torch.compile with a short run.
    max_chunks: int | None = None
    # Keys observed at the most recent pixel integration. Used to detect a
    # *new* key press (set difference) so we can snap velocity to the new
    # target instantly — overriding any momentum from prior buffered presses.
    _last_keys: set[str] = field(default_factory=set)

    def snapshot_keys(self) -> set[str]:
        with self.keys_lock:
            return set(self.keys_down)

    def integrate_pixels_up_to(self, target_pixel_exclusive: int) -> None:
        """Advance the velocity model until ``last_integrated_pixel`` reaches
        ``target_pixel_exclusive``.

        Each per-pixel step reads the *current* key state. If the user just
        pressed a new key (the current set has anything not in the previous
        integration's set), we **snap** velocity to the new target instead
        of smoothing — so a new press immediately overrides any momentum
        from previously-buffered presses. Releases (key removed from the
        set) still go through the smooth exponential coast so motion fades
        rather than snapping to zero.
        """
        dt = 1.0 / FPS
        while self.last_integrated_pixel < target_pixel_exclusive:
            keys = self.snapshot_keys()
            target = _target_velocity_from_keys(keys)
            new_keys = keys - self._last_keys
            if new_keys:
                # Instant override on press.
                self.velocity.tx = target.tx
                self.velocity.sx = target.sx
                self.velocity.yaw = target.yaw
                self.velocity.pitch = target.pitch
            else:
                self.velocity.step_toward(target, dt)
            self._last_keys = keys

            v = self.velocity
            # Pitch update with clamping.
            new_pitch = self.current_pitch + v.pitch
            new_pitch = max(-PITCH_LIMIT_RAD, min(PITCH_LIMIT_RAD, new_pitch))
            pitch_step = new_pitch - self.current_pitch
            self.current_pitch = new_pitch

            R = self.current_pose[:3, :3]
            R_new = _rot_y(v.yaw) @ R @ _rot_x(pitch_step)

            # Horizontal-plane WASD translation (project forward / right
            # onto the y=0 plane).
            fwd = R_new[:, 2].copy()
            fwd[1] = 0.0
            rgt = R_new[:, 0].copy()
            rgt[1] = 0.0
            fn = float(np.linalg.norm(fwd))
            rn = float(np.linalg.norm(rgt))
            if fn > 0:
                fwd /= fn + 1e-6
            if rn > 0:
                rgt /= rn + 1e-6
            T_ = self.current_pose[:3, 3] + fwd * v.tx + rgt * v.sx

            self.current_pose = np.eye(4, dtype=np.float64)
            self.current_pose[:3, :3] = R_new
            self.current_pose[:3, 3] = T_

            self.pixel_poses_full[self.last_integrated_pixel] = self.current_pose
            self.last_integrated_pixel += 1


# ============================================================================
# Camera-tensor in-place injection
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
    """Writes ``raymap`` + ``chunk_plucker`` slices for a range of latents.

    Designed for in-place mutation of the pre-allocated full-length tensors
    that ``SelfForcingFlowEulerCamCtrl._inject_sliced_extras`` slices per
    chunk at sample time. The sampler reads the camera tensors when each
    chunk's denoising starts, so as long as we have written the chunk's
    latent range *before* the sampler is called again, the new motion is
    picked up.
    """

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

        # Intrinsics in latent-grid pixels (constant for the session).
        intr_latent = intrinsics_pixel_vec4.astype(np.float32).copy()
        intr_latent[0] *= latent_w / float(pixel_w)
        intr_latent[2] *= latent_w / float(pixel_w)
        intr_latent[1] *= latent_h / float(pixel_h)
        intr_latent[3] *= latent_h / float(pixel_h)
        self._intr_latent = intr_latent  # (4,)

    def write_latents(
        self,
        lat_start: int,
        lat_end: int,
        pixel_poses_full: np.ndarray,
    ) -> None:
        """Re-write ``raymap_full[:, lat_start:lat_end]`` and
        ``chunk_plucker_full[:, :, lat_start:lat_end]`` from
        ``pixel_poses_full`` (already relativised to identity at frame 0)."""
        S = self.S
        n_lat = lat_end - lat_start
        device = self.raymap_full.device
        dtype = self.raymap_full.dtype

        # Gather plucker pixel windows per latent.
        # Latent k (k > 0): pixels [k*S - (S-1), k*S + 1).
        # Latent 0:         pixels [0, S).
        pose_slab = np.empty((n_lat, S, 4, 4), dtype=np.float32)
        for li, k in enumerate(range(lat_start, lat_end)):
            s = max(0, k * S - (S - 1))
            e = s + S
            slab = pixel_poses_full[s:e]
            if slab.shape[0] < S:
                pad = S - slab.shape[0]
                slab = np.concatenate(
                    [slab, np.broadcast_to(slab[-1:], (pad, 4, 4))], axis=0
                )
            pose_slab[li] = slab

        pose_t = torch.from_numpy(pose_slab).to(device=device, dtype=torch.float32)
        intr_latent_t = torch.from_numpy(
            np.broadcast_to(self._intr_latent, (n_lat, S, 4)).copy()
        ).to(device=device, dtype=torch.float32)

        # compute_raymap consumes (T, 4, 4) + (T, 4) → (T, H, W, 6).
        pose_flat = pose_t.reshape(n_lat * S, 4, 4)
        intr_flat = intr_latent_t.reshape(n_lat * S, 4)
        plucker = compute_raymap(
            intr_flat, pose_flat, self.latent_h, self.latent_w, use_plucker=True
        )  # (n_lat * S, H, W, 6)
        plucker = (
            plucker.reshape(n_lat, S, self.latent_h, self.latent_w, 6)
            .permute(0, 1, 4, 2, 3)  # (n_lat, S, 6, H, W)
            .reshape(n_lat, S * 6, self.latent_h, self.latent_w)
        )
        # chunk_plucker_full layout: (1, S*6, T_max, H, W). Write the n_lat
        # slabs into latent dim.
        self.chunk_plucker_full[0, :, lat_start:lat_end, :, :] = (
            plucker.permute(1, 0, 2, 3).to(dtype=dtype)
        )

        # raymap row for latent k = flatten(pose @ pixel k*S) ++ intr_latent.
        raymap_pixel = np.arange(lat_start, lat_end) * S
        raymap_pixel = np.minimum(raymap_pixel, pixel_poses_full.shape[0] - 1)
        raymap_poses = pixel_poses_full[raymap_pixel]  # (n_lat, 4, 4)
        raymap_poses_t = torch.from_numpy(raymap_poses.astype(np.float32)).to(
            device=device, dtype=torch.float32
        )
        intr_lat_rows = torch.from_numpy(
            np.broadcast_to(self._intr_latent, (n_lat, 4)).copy()
        ).to(device=device, dtype=torch.float32)
        raymap_rows = torch.cat(
            [raymap_poses_t.reshape(n_lat, 16), intr_lat_rows], dim=-1
        ).to(dtype=dtype)
        self.raymap_full[0, lat_start:lat_end, :] = raymap_rows


# ============================================================================
# QueueFrameSink — FrameSink implementation that pushes JPEGs to asyncio
# ============================================================================


class QueueFrameSink:
    """Drop-in sink for ``streaming_pipeline.run_streaming_inference``.

    JPEG-encodes pixel frames and posts them into the session's asyncio
    frame queue via ``loop.call_soon_threadsafe``. Drops the oldest frame
    when the queue exceeds ``SERVER_QUEUE_HIGHWATER`` so the client always
    sees the most recent output.

    Encoding runs on a dedicated background thread so the streaming
    orchestrator's main thread (which drives the next AR chunk) isn't
    blocked by JPEG work. ``write_chunk`` is therefore O(1) — it just
    hands the raw uint8 numpy chunk to the encoder thread.
    """

    import queue as _queue_mod

    def __init__(self, session: Session) -> None:
        import queue
        self._session = session
        self._closed = False
        # Raw pixel chunks pending encoding. Small bound so producer naturally
        # backpressures if the encoder somehow falls more than ~12 s behind.
        self._pixel_q: queue.Queue = queue.Queue(maxsize=8)
        self._encoder_thread = threading.Thread(
            target=self._encoder_loop,
            name="sana-wm-jpeg-encoder",
            daemon=True,
        )
        self._encoder_thread.start()

    def write_chunk(self, frames_uint8: np.ndarray) -> None:
        if self._closed:
            return
        # O(1): hand the raw uint8 chunk to the encoder thread.
        try:
            self._pixel_q.put_nowait(frames_uint8)
        except Exception:
            # Drop on overflow — encoder is far behind; better to lose a
            # chunk than to block the orchestrator.
            pass

    def close(self):
        if self._closed:
            return None
        self._closed = True
        # Signal encoder to exit; join briefly so we don't leak threads.
        try:
            self._pixel_q.put_nowait(None)
        except Exception:
            pass
        self._encoder_thread.join(timeout=2.0)
        return None

    def _encoder_loop(self) -> None:
        import queue
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
                buf = io.BytesIO()
                Image.fromarray(item[i]).save(buf, "JPEG", quality=JPEG_QUALITY)
                jpeg = buf.getvalue()
                loop = self._session.loop
                q = self._session.frame_q
                if loop is not None and q is not None:
                    loop.call_soon_threadsafe(_push_frame, q, jpeg)


class DiscardFrameSink:
    """FrameSink that drops every chunk. Used by the startup warmup."""

    def write_chunk(self, frames_uint8: np.ndarray) -> None:  # noqa: ARG002
        return

    def close(self):
        return None


def _push_frame(q: asyncio.Queue, jpeg: bytes) -> None:
    """Push a frame, dropping the oldest if we'd exceed the high-water mark."""
    if q.qsize() >= SERVER_QUEUE_HIGHWATER:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        q.put_nowait(jpeg)
    except asyncio.QueueFull:
        # qsize check above is racy under multiple producers; this branch
        # shouldn't be hit in single-producer mode, but be defensive.
        try:
            q.get_nowait()
            q.put_nowait(jpeg)
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            pass


# ============================================================================
# Pipeline & scene cache (built once at startup)
# ============================================================================


@dataclass
class SceneCache:
    """Pre-encoded per-scene tensors that don't change at runtime."""

    cond: torch.Tensor
    cond_mask: torch.Tensor
    neg: torch.Tensor
    neg_mask: torch.Tensor
    first_latent: torch.Tensor  # (1, C, 1, h, w), VAE-encoded first frame
    intrinsics_pixel_vec4: np.ndarray  # (4,) intrinsics after crop transform
    refiner_prompt_embeds: torch.Tensor | None  # cached connector embeds
    refiner_prompt_attention_mask: torch.Tensor | None


@dataclass
class LoadedPipeline:
    pipeline: SanaWMPipeline
    scene_caches: dict[str, SceneCache]


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _resolve_paths(streaming_root: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve the five required component paths under streaming_root."""
    cfg = streaming_root / "sana_wm_streaming_1600m_720p.yaml"
    model = streaming_root / "sana_dit" / "model.pt"
    causal_vae = streaming_root / "ltx2_causal_vae"
    refiner = streaming_root / "refiner_diffusers"
    gemma = streaming_root / "gemma3_12b"
    for label, p in [
        ("config", cfg), ("model", model), ("causal_vae", causal_vae),
        ("refiner", refiner), ("gemma", gemma),
    ]:
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


# ============================================================================
# Persistent disk cache for per-scene tensors
# ============================================================================
#
# Stage-1 text embeddings, the empty-prompt negative embedding, the first-
# frame VAE latents, and the refiner's connector-packed prompt embeddings
# are all deterministic functions of the prompt text, the image bytes, and
# the model identity. We persist them under
# ``~/.cache/sana_wm_streaming_demo/v<schema>/<config-sig>/`` so subsequent
# launches (including different SLURM jobs that share the same cache dir)
# never have to re-encode them. Most importantly: when all 3 scenes hit on
# the refiner side, the **11-shard Gemma3-12B text encoder is never loaded**.

_CACHE_SCHEMA_VERSION = 1
_DEFAULT_CACHE_DIRNAME = "sana_wm_streaming_demo"


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _resolve_cache_root(override: str | None = None) -> Path:
    """Resolve the cache root. Precedence: --cache_dir > $SANA_WM_STREAMING_CACHE_DIR
    > $XDG_CACHE_HOME/sana_wm_streaming_demo > ~/.cache/sana_wm_streaming_demo.
    """
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
    """Hash a fingerprint of the streaming weights. Cache is segregated by
    this so swapping the checkpoint bundle automatically invalidates the
    cache. We use file sizes + mtimes rather than full hashes — full hashes
    on 50 GB would dominate startup.
    """
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


def _persistent_cache_dir(
    streaming_root: Path, override: str | None = None
) -> Path:
    root = _resolve_cache_root(override)
    return root / f"v{_CACHE_SCHEMA_VERSION}" / _config_signature(streaming_root)


def _safe_torch_save(obj: dict, path: Path) -> bool:
    """Atomic save with a sibling .tmp file; swallow IO errors with a warning."""
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
    except Exception as exc:
        LOGGER.warning(f"[cache] load failed at {path}: {exc!r} — recomputing")
        return None


def load_pipeline(
    streaming_root: Path, device: torch.device, cache_dir: Path | None = None,
) -> LoadedPipeline:
    """One-shot: build SanaWMPipeline + compile + scene caches.

    Cold start runs torch.compile (max-autotune-no-cudagraphs) — ~3 min first
    time, ~30 s on a warm cache. Scene-tensor disk cache further skips the
    ~8 min Gemma3 cold load when its refiner entries are all on disk.
    """
    _apply_fast_defaults()
    cfg, model, causal_vae, refiner_root, gemma_root = _resolve_paths(streaming_root)
    config: InferenceConfig = pyrallis.parse(
        config_class=InferenceConfig, config_path=str(cfg), args=[]
    )
    config.vae.vae_type = "LTX2VAE_diffusers_causal"
    config.vae.vae_pretrained = str(causal_vae)

    refiner = RefinerSettings(
        root=str(refiner_root),
        gemma_root=str(gemma_root),
        sink_size=1,
        seed=42,
        block_size=NUM_FRAME_PER_BLOCK,
        kv_max_frames=11,
    )

    pipeline = SanaWMPipeline(
        config=config,
        model_path=str(model),
        device=device,
        refiner=refiner,
        offload_vae=False,
        offload_refiner=False,
        logger=LOGGER,
    )

    LOGGER.info("[streaming] compiling vae.decoder + refiner.transformer "
                "(max-autotune-no-cudagraphs)…")
    pipeline.vae.decoder = torch.compile(
        pipeline.vae.decoder, mode="max-autotune-no-cudagraphs", dynamic=True
    )
    pipeline.refiner.transformer = torch.compile(
        pipeline.refiner.transformer, mode="max-autotune-no-cudagraphs", dynamic=True
    )

    scene_caches = _build_scene_caches(pipeline, cache_dir=cache_dir)
    return LoadedPipeline(pipeline=pipeline, scene_caches=scene_caches)


def _build_scene_caches(
    pipeline: SanaWMPipeline,
    cache_dir: Path | None = None,
) -> dict[str, SceneCache]:
    """Pre-encode prompt + first-frame VAE + refiner connector embeds per scene.

    Two-pass:
      1. Try to load every entry from disk. When all three scenes' refiner
         entries hit, the Gemma3-12B encoder is **never loaded** (~8 min cold
         load saved).
      2. For any misses: load the relevant text encoders, compute, and persist.

    Schema version + per-file prompt + image hashes guarantee that edits to
    asset files invalidate the cache cleanly.
    """
    if cache_dir is not None:
        LOGGER.info(f"[scenes] persistent cache dir: {cache_dir}")

    refiner = pipeline.refiner
    device = pipeline.device

    # Per-scene scratch (everything but the SceneCache dataclass — that's
    # assembled at the end).
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

        # Cropped first-frame image + cropped intrinsics — cheap, always recompute.
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

        # Stage-1 disk attempt.
        s1_path = (cache_dir / f"stage1_{scene.id}.pt") if cache_dir is not None else None
        loaded = _try_load(s1_path, device) if s1_path is not None else None
        if (
            loaded is not None
            and loaded.get("schema") == _CACHE_SCHEMA_VERSION
            and loaded.get("prompt_hash") == prompt_hash
            and loaded.get("image_hash") == image_hash
        ):
            sc_scratch["cond"] = loaded["cond"].to(device).contiguous()
            sc_scratch["cond_mask"] = loaded["cond_mask"].to(device).contiguous()
            sc_scratch["first_latent"] = loaded["first_latent"].to(device).contiguous()
        else:
            stage1_misses.append(scene)

        # Refiner disk attempt.
        r_path = (cache_dir / f"refiner_{scene.id}.pt") if cache_dir is not None else None
        loaded = _try_load(r_path, refiner.device) if r_path is not None else None
        if (
            loaded is not None
            and loaded.get("schema") == _CACHE_SCHEMA_VERSION
            and loaded.get("prompt_hash") == prompt_hash
        ):
            sc_scratch["refiner_embeds"] = loaded["embeds"].to(
                device=refiner.device, dtype=refiner.dtype
            ).contiguous()
            sc_scratch["refiner_mask"] = loaded["mask"].to(refiner.device).contiguous()
        else:
            refiner_misses.append(scene)

        scratch[scene.id] = sc_scratch

    LOGGER.info(
        f"[scenes] disk hits: stage1={len(SCENES) - len(stage1_misses)}/{len(SCENES)}, "
        f"refiner={len(SCENES) - len(refiner_misses)}/{len(SCENES)}, "
        f"neg={'hit' if neg_loaded is not None else 'miss'}"
    )

    # The diffusers refiner loads transformer + connectors to CPU by default
    # (``from_pretrained`` ignores ``device``). The bidirectional code path
    # gets them onto GPU as a side-effect of the Gemma-encode dance in
    # ``refine_latents``; we bypass that path, so we must move them ourselves
    # unconditionally — including when refiner caches are fully on disk.
    refiner.transformer.to(refiner.device)
    refiner.connectors.to(refiner.device)

    # --- Compute stage-1 misses ----------------------------------------
    with torch.inference_mode():
        if stage1_misses or neg_loaded is None:
            for scene in stage1_misses or SCENES[:1]:  # at least one scene to fill neg
                prompt = scratch[scene.id]["prompt"]
                cond, cond_mask, neg_t, neg_mask_t = pipeline._encode_prompts(prompt, "")
                if neg_loaded is None:
                    neg_loaded = (neg_t.contiguous(), neg_mask_t.contiguous())
                    if neg_path is not None:
                        _safe_torch_save(
                            {
                                "schema": _CACHE_SCHEMA_VERSION,
                                "neg": neg_t.detach().cpu(),
                                "neg_mask": neg_mask_t.detach().cpu(),
                            },
                            neg_path,
                        )
                if scene in stage1_misses:
                    cropped = scratch[scene.id]["cropped"]
                    img = (T.ToTensor()(cropped) * 2.0 - 1.0).unsqueeze(0).unsqueeze(2)
                    first_latent = vae_encode(
                        pipeline.config.vae.vae_type,
                        pipeline.vae,
                        img.to(device, dtype=pipeline.vae_dtype),
                        device=device,
                    ).to(pipeline.weight_dtype)
                    scratch[scene.id]["cond"] = cond.contiguous()
                    scratch[scene.id]["cond_mask"] = cond_mask.contiguous()
                    scratch[scene.id]["first_latent"] = first_latent.contiguous()
                    s1_path = (cache_dir / f"stage1_{scene.id}.pt") if cache_dir is not None else None
                    if s1_path is not None:
                        _safe_torch_save(
                            {
                                "schema": _CACHE_SCHEMA_VERSION,
                                "prompt_hash": scratch[scene.id]["prompt_hash"],
                                "image_hash": scratch[scene.id]["image_hash"],
                                "cond": cond.detach().cpu(),
                                "cond_mask": cond_mask.detach().cpu(),
                                "first_latent": first_latent.detach().cpu(),
                            },
                            s1_path,
                        )

        # --- Compute refiner misses (load Gemma3 ONLY if needed) -------
        if refiner_misses:
            LOGGER.info(
                f"[scenes] refiner cache miss for "
                f"{[s.id for s in refiner_misses]} — loading Gemma3 once "
                f"(~8 min cold, ~30 s warm)…"
            )
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
                    [prompt], padding="max_length",
                    max_length=refiner.text_max_sequence_length,
                    truncation=True, add_special_tokens=True, return_tensors="pt",
                )
                input_ids = text_inputs.input_ids.to(refiner.device)
                attn = text_inputs.attention_mask.to(refiner.device)
                outputs = text_backbone(
                    input_ids=input_ids, attention_mask=attn, output_hidden_states=True
                )
                hidden = torch.stack(outputs.hidden_states, dim=-1)
                seqlens = attn.sum(dim=-1)
                prompt_embeds_packed = _pack_text_embeds(
                    hidden, seqlens, device=refiner.device, padding_side=tok.padding_side
                ).to(dtype=refiner.dtype)
                conn_embeds, _, conn_mask = refiner.connectors(prompt_embeds_packed, attn)
                scratch[scene.id]["refiner_embeds"] = conn_embeds.to(
                    device=refiner.device, dtype=refiner.dtype
                ).contiguous()
                scratch[scene.id]["refiner_mask"] = conn_mask.to(refiner.device).contiguous()
                r_path = (cache_dir / f"refiner_{scene.id}.pt") if cache_dir is not None else None
                if r_path is not None:
                    _safe_torch_save(
                        {
                            "schema": _CACHE_SCHEMA_VERSION,
                            "prompt_hash": scratch[scene.id]["prompt_hash"],
                            "embeds": conn_embeds.detach().cpu(),
                            "mask": conn_mask.detach().cpu(),
                        },
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

    # --- Assemble SceneCache objects -----------------------------------
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


# ============================================================================
# Session loop (producer thread)
# ============================================================================


def warmup(loaded: LoadedPipeline, passes: int = 2) -> None:
    """Prime ``torch.compile`` by running ``passes`` full dummy sessions.

    A single pass eliminates the ~3 min cold ``torch.compile`` cost, but
    leaves a residual ~5 s penalty on the *next* session's chunk 0
    (Inductor finalises a few specialisations, cuDNN picks algorithms,
    CUDA streams get cached). A second pass burns that off in private so
    the first user click is already at steady-state speed.

    Each pass uses ``SCENES[0]`` and ``DiscardFrameSink`` — no MP4 /
    WebSocket setup involved. Velocity is held at zero, exercising the
    same shape envelope a real session uses.

    Expected cost on H100:
      * pass 1: ~165 s (cold compile + 13 chunks of work)
      * pass 2: ~35 s (residual specialisations + 13 chunks of work)
    """
    LOGGER.info(f"[warmup] running {passes} full session(s) to prime torch.compile…")
    t_total = time.time()
    for i in range(passes):
        t0 = time.time()
        warm_session = Session(scene=SCENES[0])
        warm_session.fire_event.set()  # skip the armed→fire wait
        try:
            _run_session_inner(loaded, warm_session, sink=DiscardFrameSink())
        except Exception:
            LOGGER.exception(
                f"[warmup] pass {i + 1}/{passes} failed "
                "(interactive sessions may pay residual cold cost)"
            )
            break
        finally:
            warm_session.finished_event.set()
        LOGGER.info(f"[warmup] pass {i + 1}/{passes} done in {time.time() - t0:.1f}s")
    LOGGER.info(
        f"[warmup] total {time.time() - t_total:.1f}s — interactive sessions are now warm"
    )


def run_session(
    loaded: LoadedPipeline, session: Session, sink: "FrameSinkLike | None" = None,
) -> None:
    """Drive the streaming pipeline for one interactive session.

    Runs on a background thread. Sets ``session.finished_event`` when done.
    ``sink`` defaults to a ``QueueFrameSink`` writing JPEGs to ``session.frame_q``;
    pass ``DiscardFrameSink()`` for a warmup run.
    """
    try:
        _run_session_inner(loaded, session, sink=sink)
    except Exception:
        LOGGER.exception("[session] producer thread crashed")
    finally:
        session.finished_event.set()


# Lightweight alias — the streaming_pipeline.FrameSink Protocol is structural
# so any object with ``write_chunk`` + ``close`` qualifies.
FrameSinkLike = object


def _run_session_inner(
    loaded: LoadedPipeline,
    session: Session,
    sink: "FrameSinkLike | None" = None,
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

    # --- Pre-allocate the full camera tensors and seed identity ----------
    # raymap shape: (1, T_lat, 20). chunk_plucker: (1, S*6, T_lat, H_lat, W_lat).
    raymap_dim = 16 + 4  # flattened pose + intrinsics_vec4
    S = VAE_TIME_STRIDE
    raymap_full = torch.empty(
        1, latent_T, raymap_dim, device=device, dtype=pipeline.weight_dtype
    )
    chunk_plucker_full = torch.empty(
        1, S * 6, latent_T, latent_h, latent_w,
        device=device, dtype=pipeline.weight_dtype,
    )

    # Pixel-pose array — single source of truth.
    pixel_poses_full = np.tile(np.eye(4, dtype=np.float64)[None], (n_pixel_total, 1, 1))
    session.pixel_poses_full = pixel_poses_full
    session.last_integrated_pixel = 1  # pixel 0 stays identity

    injector = CameraInjector(
        raymap_full=raymap_full,
        chunk_plucker_full=chunk_plucker_full,
        intrinsics_pixel_vec4=sc.intrinsics_pixel_vec4,
        pixel_h=TARGET_HEIGHT,
        pixel_w=TARGET_WIDTH,
        latent_h=latent_h,
        latent_w=latent_w,
    )

    chunk_indices = _build_chunk_indices(latent_T, NUM_FRAME_PER_BLOCK)
    n_chunks = len(chunk_indices) - 1

    # Pre-seed chunk 0's camera (pixels [0, ef0*S) all identity).
    injector.write_latents(0, chunk_indices[1], pixel_poses_full)

    # --- Build solver and the camera-injection wrapping iterator ---------
    cfg_scale = 1.0
    cond = sc.cond
    neg = sc.neg
    cond_mask = sc.cond_mask
    raymap_for_model = raymap_full
    chunk_plucker_for_model = chunk_plucker_full
    mask_cfg = cond_mask  # cfg_scale == 1 (distilled)

    chunk_index_extra = get_chunk_index_from_config(config, num_frames=latent_T)
    model_kwargs: dict = dict(
        data_info={
            "img_hw": torch.tensor(
                [[TARGET_HEIGHT, TARGET_WIDTH]], dtype=torch.float, device=device
            ),
            "condition_frame_info": {0: 0.0},
        },
        mask=mask_cfg,
        camera_conditions=raymap_for_model,
        chunk_plucker=chunk_plucker_for_model,
    )
    if chunk_index_extra is not None:
        model_kwargs["chunk_index"] = chunk_index_extra

    flow_shift = pipeline._resolve_flow_shift(8.0)
    solver = SelfForcingFlowEulerCamCtrl(
        pipeline.model,
        condition=cond,
        uncondition=neg,
        cfg_scale=cfg_scale,
        flow_shift=flow_shift,
        model_kwargs=model_kwargs,
        base_chunk_frames=NUM_FRAME_PER_BLOCK,
        num_cached_blocks=2,
        sink_token=True,
        use_softmax_attention=True,
    )

    # --- Initial latent tensor ------------------------------------------
    generator = torch.Generator(device=device).manual_seed(42)
    z = torch.randn(
        1, sc.first_latent.shape[1], latent_T, latent_h, latent_w,
        dtype=pipeline.weight_dtype, device=device, generator=generator,
    )
    z[:, :, :1] = sc.first_latent

    # ``yield_save_separately=True`` matches the streaming_pipeline.split_kv_save
    # default — each chunk produces two yields: the clean chunk, then a ``None``
    # sentinel after the KV-save pass. The orchestrator calls ``next()`` twice
    # per chunk; our wrapping iterator must forward both.
    inner_iter = solver.sample_chunks(
        z,
        generator=generator,
        denoising_step_list=list(DEFAULT_DENOISING_STEP_LIST),
        yield_save_separately=True,
    )

    def injected_iter() -> Iterator[tuple[int, torch.Tensor, int, int] | None]:
        """Wraps inner iterator; forwards chunk + sentinel yields and updates
        the next chunk's camera tensors between the sentinel and the next
        chunk's denoising start."""
        last_k = -1
        for item in inner_iter:
            if session.stop_event.is_set():
                return
            if item is None:
                # KV-save sentinel for chunk ``last_k``. Forward to orchestrator.
                yield None
                # Now write camera for chunk last_k + 1 BEFORE the inner
                # generator is asked for the next item (which would start
                # chunk last_k + 1's denoising and read the camera slice).
                next_k = last_k + 1
                if next_k < n_chunks:
                    next_sf = chunk_indices[next_k]
                    next_ef = chunk_indices[next_k + 1]
                    target_pixel_excl = min(
                        n_pixel_total, (next_ef - 1) * S + S
                    )
                    session.integrate_pixels_up_to(target_pixel_excl)
                    injector.write_latents(next_sf, next_ef, pixel_poses_full)
                continue
            k, latent_view, sf, ef = item
            last_k = k
            session.n_chunks_emitted = k + 1
            yield (k, latent_view, sf, ef)
            if (
                session.max_chunks is not None
                and session.n_chunks_emitted >= session.max_chunks
            ):
                session.stop_event.set()
                return

    # --- Refiner runner + VAE streaming decoder --------------------------
    refiner = pipeline.refiner
    sigmas_t = torch.tensor(DEFAULT_REFINER_SIGMAS, dtype=torch.float32, device=device)
    refiner_runner = RefinerChunkRunner(
        refiner,
        prompt_embeds=sc.refiner_prompt_embeds,
        prompt_attention_mask=sc.refiner_prompt_attention_mask,
        fps=float(FPS),
        sigmas=sigmas_t,
        source_sink_frames=1,
        block_size=NUM_FRAME_PER_BLOCK,
        kv_max_frames=11,
        seed=42,
        spatial_shape=(latent_h, latent_w),
    )
    vae_streaming_decoder = CausalVaeStreamingDecoder(pipeline.vae)

    if sink is None:
        sink = QueueFrameSink(session)
    cfg = StreamingPipelineConfig(
        sink_size=1,
        block_size=NUM_FRAME_PER_BLOCK,
        fps=FPS,
        output_path="/tmp/sana_wm_streaming_unused.mp4",  # unused with sink
        mp4_crf=18,
        mp4_preset="medium",
        drop_first_pixel=True,
    )

    # All Python-side setup is done; signal "armed" and park here until the
    # client presses a steering key (sets ``fire_event``) or aborts. The
    # control channel watches ``armed_event`` and forwards an ``armed`` event
    # to the browser so the UI can swap from "preparing" to "ready".
    session.armed_event.set()
    LOGGER.info(f"[session] {scene.id} armed; waiting for fire")
    while not session.fire_event.is_set() and not session.stop_event.is_set():
        if session.fire_event.wait(timeout=0.25):
            break
    if session.stop_event.is_set():
        LOGGER.info(f"[session] {scene.id} aborted before fire")
        return

    LOGGER.info(f"[session] starting {scene.id} (latent_T={latent_T}, n_chunks={n_chunks})")
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
            sink=sink,
            logger=LOGGER,
        )
    finally:
        sink.close()
    LOGGER.info(f"[session] done in {time.time() - t0:.1f}s")


# ============================================================================
# FastAPI app
# ============================================================================


LOADED: LoadedPipeline | None = None
SESSION_LOCK = asyncio.Lock()
CURRENT_SESSION: Session | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global LOADED
    streaming_root = Path(app.state.streaming_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir_override = getattr(app.state, "cache_dir", None)
    cache_dir = _persistent_cache_dir(streaming_root, cache_dir_override)
    LOGGER.info(f"[startup] loading streaming pipeline from {streaming_root}…")
    t0 = time.time()
    LOADED = load_pipeline(streaming_root, device, cache_dir=cache_dir)
    LOGGER.info(f"[startup] pipeline loaded in {time.time() - t0:.1f}s")

    if getattr(app.state, "do_warmup", True):
        warmup(LOADED)
    else:
        LOGGER.warning("[startup] --no_warmup: first session will pay the cold compile cost")

    LOGGER.info("[startup] ready — open the page in a browser")
    yield


# ----- Inlined client HTML -------------------------------------------------
# Embedded here so the entire demo is a single Python file (no app/static/).
# The canvas + WebSocket logic is intentionally framework-free vanilla JS.

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=1500" />
  <title>SANA-WM Interactive Streaming</title>
  <style>
    :root {
      --bg: #0c0c10;
      --panel: #16161c;
      --line: #2a2a35;
      --fg: #e6e6ea;
      --dim: #8b8b94;
      --green: #a4d65e;
      --green-mid: #76b900;
      --yellow: #e5c050;
      --red: #ff5a52;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0; padding: 0; background: var(--bg); color: var(--fg);
      font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace;
      font-size: 13px; line-height: 1.45;
    }
    header {
      padding: 14px 22px; border-bottom: 1px solid var(--line);
      display: flex; align-items: baseline; gap: 18px;
    }
    header h1 { margin: 0; font-size: 16px; font-weight: 600; }
    header .meta { color: var(--dim); font-size: 12px; }
    main {
      display: grid;
      grid-template-columns: minmax(960px, 1fr) 360px;
      gap: 22px; padding: 22px;
    }
    .video-card {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 10px; padding: 16px;
    }
    canvas {
      display: block;
      width: 100%;
      max-width: 1280px;
      aspect-ratio: 1280 / 704;
      background: #000;
      border-radius: 6px;
    }
    .video-stage {
      position: relative;
      width: 100%;
      max-width: 1280px;
      aspect-ratio: 1280 / 704;
    }
    .video-stage canvas { width: 100%; height: 100%; }
    .preview {
      position: absolute; inset: 0;
      width: 100%; height: 100%;
      object-fit: cover; border-radius: 6px;
      pointer-events: none;
      opacity: 0; transition: opacity 0.18s ease-out;
    }
    .preview.visible { opacity: 1; }
    .overlay {
      position: absolute; inset: 0;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      gap: 14px;
      pointer-events: none; border-radius: 6px;
      background: rgba(12, 12, 16, 0.55);
      opacity: 0; transition: opacity 0.18s ease-out;
    }
    .overlay.visible { opacity: 1; }
    .spinner {
      width: 42px; height: 42px; border-radius: 50%;
      border: 3px solid rgba(255, 255, 255, 0.18);
      border-top-color: var(--green);
      animation: spin 0.9s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .overlay-text {
      color: var(--fg); font-size: 13px; font-weight: 500;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
      letter-spacing: 0.02em; text-align: center;
    }
    .overlay-sub {
      color: var(--dim); font-size: 11px; margin-top: -6px;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
    }
    .side { display: flex; flex-direction: column; gap: 18px; }
    .panel {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 10px; padding: 14px 16px;
    }
    .panel h2 {
      margin: 0 0 10px; font-size: 12px; font-weight: 600;
      letter-spacing: 0.08em; text-transform: uppercase; color: var(--dim);
    }
    .scenes { display: flex; flex-direction: column; gap: 6px; }
    .scene-btn {
      background: #1d1d24; border: 1px solid var(--line); color: var(--fg);
      padding: 8px 10px; text-align: left; border-radius: 6px; cursor: pointer;
      font: inherit;
    }
    .scene-btn:hover { border-color: var(--green-mid); }
    .scene-btn.selected {
      border-color: var(--green); background: rgba(164, 214, 94, 0.08);
    }
    .actions { display: flex; gap: 8px; }
    .btn {
      flex: 1; padding: 9px 12px; border-radius: 6px; cursor: pointer;
      font: inherit; border: 1px solid var(--line); background: #1d1d24;
      color: var(--fg);
    }
    .btn.primary {
      background: var(--green-mid); border-color: var(--green-mid); color: #0c0c10;
      font-weight: 600;
    }
    .btn.primary:hover { background: var(--green); border-color: var(--green); }
    .btn:disabled { opacity: 0.45; cursor: not-allowed; }
    .keypad {
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px;
    }
    .key {
      background: #1d1d24; border: 1px solid var(--line); border-radius: 6px;
      padding: 12px 0; text-align: center; user-select: none; font-weight: 600;
      cursor: pointer; color: var(--fg);
    }
    .key.active {
      background: var(--green-mid); border-color: var(--green-mid);
      color: #0c0c10;
    }
    .key.spacer { background: transparent; border-color: transparent; cursor: default; }
    .pad-label {
      font-size: 11px; color: var(--dim); margin: 6px 0 4px;
      letter-spacing: 0.06em; text-transform: uppercase;
    }
    .statusbar {
      display: flex; gap: 14px; align-items: center;
      padding: 8px 14px; background: var(--panel);
      border: 1px solid var(--line); border-radius: 6px;
      font-size: 12px; color: var(--dim);
    }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--dim); }
    .dot.green { background: var(--green); }
    .dot.yellow { background: var(--yellow); }
    .dot.red { background: var(--red); }
    .hint {
      color: var(--dim); font-size: 11px; line-height: 1.55; margin-top: 6px;
    }
    .kbd {
      display: inline-block; padding: 1px 6px; background: #1d1d24;
      border: 1px solid var(--line); border-radius: 4px; color: var(--fg);
      font-size: 11px;
    }
  </style>
</head>
<body>
  <header>
    <h1>SANA-WM · Interactive Streaming</h1>
    <span class="meta">19.5 s max · 16 fps · H.264-equivalent live preview</span>
  </header>

  <main>
    <section class="video-card">
      <div class="video-stage">
        <canvas id="video" width="1280" height="704"></canvas>
        <img id="preview" class="preview" alt="" />
        <div id="overlay" class="overlay">
          <div id="spinner" class="spinner"></div>
          <div id="overlay-text" class="overlay-text">preparing…</div>
          <div id="overlay-sub" class="overlay-sub"></div>
        </div>
      </div>
      <div class="statusbar" style="margin-top: 12px;">
        <span class="dot" id="dot"></span>
        <span id="status">disconnected</span>
        <span style="flex: 1"></span>
        <span id="timer">0.0 / 19.5 s</span>
        <span id="buf">buf=0</span>
      </div>
      <div class="hint">
        Driving-style controls. <span class="kbd">W</span> / <span class="kbd">S</span> = forward / back ·
        <span class="kbd">A</span> / <span class="kbd">D</span> = steer left / right.
        Hold <span class="kbd">W</span>+<span class="kbd">D</span> to walk forward in a right arc; release everything
        and the camera coasts to a smooth stop. First frame appears ~4 s after
        the first keystroke.
      </div>
    </section>

    <aside class="side">
      <div class="panel">
        <h2>Scene</h2>
        <div class="scenes" id="scenes"></div>
        <div class="actions" style="margin-top: 12px;">
          <button class="btn primary" id="btn-start">Start</button>
          <button class="btn" id="btn-reset" disabled>Reset</button>
        </div>
      </div>

      <div class="panel">
        <h2>Control pad</h2>
        <div class="pad-label">W / S = forward / back · A / D = steer left / right</div>
        <div class="keypad">
          <div class="key spacer"></div>
          <div class="key" data-key="w">W</div>
          <div class="key spacer"></div>
          <div class="key" data-key="a">A</div>
          <div class="key" data-key="s">S</div>
          <div class="key" data-key="d">D</div>
        </div>
        <div class="hint" style="margin-top: 10px;">
          Click & hold a key cell, or use the physical keyboard. Multiple
          keys combine.
        </div>
      </div>
    </aside>
  </main>

  <script>
  (() => {
    const $ = (id) => document.getElementById(id);
    const canvas = $("video");
    const ctx = canvas.getContext("2d");
    const dot = $("dot");
    const statusEl = $("status");
    const timerEl = $("timer");
    const bufEl = $("buf");
    const startBtn = $("btn-start");
    const resetBtn = $("btn-reset");
    const previewEl = $("preview");
    const overlayEl = $("overlay");
    const spinnerEl = $("spinner");
    const overlayTextEl = $("overlay-text");
    const overlaySubEl = $("overlay-sub");

    const ALLOWED_KEYS = new Set(["w", "a", "s", "d", "i", "j", "k", "l"]);
    const FPS = 16;
    const PREROLL_FRAMES = 4;    // ~0.25 s before playback starts
    const HARD_BUFFER_CAP = 96;  // safety bound

    let ctrlWs = null;
    let frameWs = null;
    let scenes = [];
    let selectedScene = null;
    let keysDown = new Set();
    let lastSent = "";
    let sessionState = "idle";
    let running = false;
    let armed = false;
    let firstFrameSeen = false;
    let buffer = [];
    let playbackStarted = false;
    let playbackTimer = null;
    let frameCount = 0;
    let startedAt = 0;

    function setStatus(s) { statusEl.textContent = s; }
    function setDot(cls) {
      dot.classList.remove("green", "yellow", "red");
      if (cls) dot.classList.add(cls);
    }
    function showOverlay(text, sub, withSpinner) {
      overlayTextEl.textContent = text;
      overlaySubEl.textContent = sub || "";
      spinnerEl.style.display = withSpinner ? "" : "none";
      overlayEl.classList.add("visible");
    }
    function hideOverlay() {
      overlayEl.classList.remove("visible");
    }
    function showPreview(sceneId) {
      previewEl.src = `/scenes/${encodeURIComponent(sceneId)}/preview`;
      previewEl.classList.add("visible");
    }
    function hidePreview() {
      previewEl.classList.remove("visible");
      previewEl.src = "";
    }

    async function loadScenes() {
      const r = await fetch("/scenes");
      scenes = await r.json();
      const el = $("scenes");
      el.innerHTML = "";
      scenes.forEach((s, i) => {
        const b = document.createElement("button");
        b.className = "scene-btn" + (i === 0 ? " selected" : "");
        b.textContent = s.label;
        b.dataset.id = s.id;
        b.onclick = () => {
          document.querySelectorAll(".scene-btn").forEach(x => x.classList.remove("selected"));
          b.classList.add("selected");
          selectedScene = s.id;
        };
        el.appendChild(b);
      });
      selectedScene = scenes[0]?.id ?? null;
    }

    function updateKeyVisuals() {
      document.querySelectorAll(".key[data-key]").forEach(el => {
        const k = el.dataset.key;
        el.classList.toggle("active", keysDown.has(k));
      });
    }
    function sendKeys() {
      const arr = Array.from(keysDown).sort();
      const sig = arr.join(",");
      if (sig === lastSent) return;
      lastSent = sig;
      if (ctrlWs && ctrlWs.readyState === 1) {
        ctrlWs.send(JSON.stringify({ op: "keys", keys: arr }));
      }
    }
    function setKey(k, down) {
      if (!ALLOWED_KEYS.has(k)) return;
      const before = keysDown.size;
      if (down) keysDown.add(k); else keysDown.delete(k);
      if (down || before !== keysDown.size) {
        updateKeyVisuals();
        sendKeys();
        if (down && armed && !running) {
          onFire();
        }
      }
    }
    window.addEventListener("keydown", (e) => {
      if (!armed && !running) return;
      const k = e.key.toLowerCase();
      if (!ALLOWED_KEYS.has(k)) return;
      e.preventDefault();
      setKey(k, true);
    });
    window.addEventListener("keyup", (e) => {
      const k = e.key.toLowerCase();
      if (!ALLOWED_KEYS.has(k)) return;
      e.preventDefault();
      setKey(k, false);
    });
    document.querySelectorAll(".key[data-key]").forEach(el => {
      const k = el.dataset.key;
      el.addEventListener("pointerdown", (e) => { e.preventDefault(); setKey(k, true); el.setPointerCapture(e.pointerId); });
      el.addEventListener("pointerup",   (e) => { e.preventDefault(); setKey(k, false); });
      el.addEventListener("pointerleave",(e) => { setKey(k, false); });
      el.addEventListener("pointercancel",(e)=> { setKey(k, false); });
    });
    window.addEventListener("blur", () => {
      if (keysDown.size > 0) {
        keysDown.clear();
        updateKeyVisuals();
        sendKeys();
      }
    });

    function pushFrame(blob) {
      buffer.push(blob);
      while (buffer.length > HARD_BUFFER_CAP) buffer.shift();
      updateBufStatus();
      if (!firstFrameSeen) {
        firstFrameSeen = true;
        hidePreview();
      }
      if (!playbackStarted && buffer.length >= PREROLL_FRAMES) {
        playbackStarted = true;
        startPlayback();
      }
    }
    function updateBufStatus() {
      bufEl.textContent = "buf=" + buffer.length;
      if (buffer.length >= 6) setDot("green");
      else if (buffer.length >= 3) setDot("yellow");
      else if (running) setDot("red");
    }
    function startPlayback() {
      if (playbackTimer) clearInterval(playbackTimer);
      playbackTimer = setInterval(drawNext, 1000 / FPS);
      hideOverlay();
    }
    function drawNext() {
      if (buffer.length === 0) {
        // Underrun — keep the timer running and freeze the last drawn frame.
        updateBufStatus();
        return;
      }
      const blob = buffer.shift();
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);
      };
      img.src = url;
      frameCount++;
      const elapsed = frameCount / FPS;
      timerEl.textContent = elapsed.toFixed(1) + " / 19.5 s";
      updateBufStatus();
    }

    function onFire() {
      if (running) return;
      armed = false;
      running = true;
      sessionState = "streaming";
      startedAt = performance.now();
      setStatus("streaming · " + selectedScene);
      showOverlay("warming up…", "first frame in ~4 s", true);
    }

    async function start() {
      if (!selectedScene) { setStatus("pick a scene"); return; }
      startBtn.disabled = true;
      buffer = []; frameCount = 0; playbackStarted = false;
      firstFrameSeen = false;
      armed = false; running = false;
      sessionState = "preparing";
      if (playbackTimer) { clearInterval(playbackTimer); playbackTimer = null; }
      setStatus("connecting…"); setDot(null);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      showPreview(selectedScene);
      showOverlay("preparing…", "loading session state", true);

      ctrlWs = new WebSocket((location.protocol === "https:" ? "wss" : "ws") + "://" + location.host + "/ctrl");
      ctrlWs.onopen = () => {
        ctrlWs.send(JSON.stringify({ op: "start", scene: selectedScene }));
      };
      ctrlWs.onmessage = (ev) => {
        let msg; try { msg = JSON.parse(ev.data); } catch { return; }
        if (msg.event === "started") {
          setStatus("preparing · " + msg.scene);
          resetBtn.disabled = false;
          openFrameSocket();
        } else if (msg.event === "armed") {
          armed = true;
          sessionState = "armed";
          setStatus("ready · press a key to begin");
          showOverlay("press a steering key to begin",
                      "W / A / S / D", false);
        } else if (msg.event === "finished") {
          setStatus("finished");
          stopSession(false);
        } else if (msg.event === "error") {
          setStatus("error: " + msg.msg);
          stopSession(false);
        }
      };
      ctrlWs.onclose = () => {
        if (running || armed) { setStatus("disconnected"); stopSession(false); }
      };
      ctrlWs.onerror = () => { setStatus("ctrl error"); };
    }

    function openFrameSocket() {
      frameWs = new WebSocket((location.protocol === "https:" ? "wss" : "ws") + "://" + location.host + "/frames");
      frameWs.binaryType = "blob";
      frameWs.onmessage = (ev) => {
        if (typeof ev.data === "string") {
          let msg; try { msg = JSON.parse(ev.data); } catch { return; }
          if (msg.event === "finished") {
            setStatus("finished");
            stopSession(false);
          }
        } else {
          pushFrame(ev.data);
        }
      };
      frameWs.onclose = () => {
        if (running) setStatus("frames disconnected");
      };
    }

    function stopSession(sendStop = true) {
      running = false;
      armed = false;
      sessionState = "idle";
      keysDown.clear();
      updateKeyVisuals();
      if (sendStop && ctrlWs && ctrlWs.readyState === 1) {
        try { ctrlWs.send(JSON.stringify({ op: "stop" })); } catch {}
      }
      try { ctrlWs && ctrlWs.close(); } catch {}
      try { frameWs && frameWs.close(); } catch {}
      ctrlWs = null; frameWs = null;
      startBtn.disabled = false;
      resetBtn.disabled = true;
      setDot(null);
      hideOverlay();
      hidePreview();
    }

    resetBtn.addEventListener("click", () => stopSession(true));
    startBtn.addEventListener("click", start);

    (async () => {
      try { await loadScenes(); setStatus("ready"); }
      catch (e) { setStatus("scene load failed: " + e); }
    })();
  })();
  </script>
</body>
</html>
"""


app = FastAPI(lifespan=lifespan, title="SANA-WM Interactive Streaming")


@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/scenes")
async def scenes():
    return JSONResponse([{"id": s.id, "label": s.label} for s in SCENES])


@app.get("/scenes/{scene_id}/preview")
async def scene_preview(scene_id: str):
    """First-frame PNG for the scene. The UI shows this as a placeholder
    during the ``preparing`` → ``armed`` window so the user has something
    to look at while the producer builds the solver / refiner / decoder."""
    scene = SCENE_BY_ID.get(scene_id)
    if scene is None:
        return JSONResponse({"detail": f"unknown scene {scene_id!r}"}, status_code=404)
    return FileResponse(str(scene.image_path), media_type="image/png")


@app.websocket("/ctrl")
async def ws_ctrl(ws: WebSocket):
    """Control channel.

    Protocol (JSON messages):
      client -> {"op": "start", "scene": "demo_0"}
      client -> {"op": "keys",  "keys": ["w", "l"]}
      client -> {"op": "fire"}
      client -> {"op": "stop"}
      server -> {"event": "started"}   — producer thread launched
      server -> {"event": "armed"}     — solver/refiner/decoder built; waiting for fire
      server -> {"event": "finished"}
      server -> {"event": "error",   "msg": "..."}

    Either an explicit ``fire`` op or the first non-empty ``keys`` payload
    releases the armed producer to begin generation.
    """
    global CURRENT_SESSION
    await ws.accept()

    if SESSION_LOCK.locked():
        await ws.send_json({"event": "error", "msg": "another session is running"})
        await ws.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    async with SESSION_LOCK:
        session: Session | None = None
        producer_thread: threading.Thread | None = None
        armed_task: asyncio.Task | None = None
        loop = asyncio.get_running_loop()

        async def _send_armed_when_ready(s: Session) -> None:
            """Forward the producer's ``armed_event`` to the client as an
            ``armed`` JSON event. Exits silently if the session aborts before
            arming."""
            while not s.armed_event.is_set():
                if s.stop_event.is_set() or s.finished_event.is_set():
                    return
                await asyncio.sleep(0.1)
            try:
                await ws.send_json({"event": "armed", "scene": s.scene.id})
            except Exception:
                pass

        try:
            while True:
                msg = await ws.receive_json()
                op = msg.get("op")
                if op == "start":
                    scene_id = str(msg.get("scene", "demo_0"))
                    scene = SCENE_BY_ID.get(scene_id)
                    if scene is None:
                        await ws.send_json({"event": "error", "msg": f"unknown scene {scene_id!r}"})
                        continue
                    if session is not None and not session.finished_event.is_set():
                        await ws.send_json({"event": "error", "msg": "session already started"})
                        continue
                    if LOADED is None:
                        await ws.send_json({"event": "error", "msg": "pipeline not loaded"})
                        continue
                    session = Session(scene=scene)
                    session.loop = loop
                    session.frame_q = asyncio.Queue(maxsize=SERVER_QUEUE_HIGHWATER)
                    CURRENT_SESSION = session
                    producer_thread = threading.Thread(
                        target=run_session, args=(LOADED, session),
                        name=f"sana-wm-producer-{scene_id}", daemon=True,
                    )
                    producer_thread.start()
                    await ws.send_json({"event": "started", "scene": scene_id})
                    armed_task = asyncio.create_task(_send_armed_when_ready(session))
                elif op == "keys":
                    if session is None:
                        continue
                    keys = msg.get("keys", [])
                    if not isinstance(keys, list):
                        continue
                    new = {str(k).lower() for k in keys if str(k).lower() in "wasdijkl"}
                    with session.keys_lock:
                        session.keys_down = new
                    if new and not session.fire_event.is_set():
                        session.fire_event.set()
                elif op == "fire":
                    if session is not None and not session.fire_event.is_set():
                        session.fire_event.set()
                elif op == "stop":
                    if session is not None:
                        session.stop_event.set()
                        # Don't await join here — let the producer wind down
                        # at the next chunk boundary; the client will see
                        # remaining queued frames and then a "finished" event.
                else:
                    await ws.send_json({"event": "error", "msg": f"unknown op {op!r}"})
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            LOGGER.exception("[ctrl] error")
            try:
                await ws.send_json({"event": "error", "msg": repr(exc)})
            except Exception:
                pass
        finally:
            if session is not None:
                session.stop_event.set()
                # Unblock the producer if it's still parked waiting for fire.
                session.fire_event.set()
                # Let producer wind down out-of-band; the frame WS will close
                # when finished_event fires.
            if armed_task is not None and not armed_task.done():
                armed_task.cancel()
            CURRENT_SESSION = None


@app.websocket("/frames")
async def ws_frames(ws: WebSocket):
    """Server -> client video frame stream. Paces at 16 fps."""
    await ws.accept()
    period = 1.0 / FPS
    # Wait briefly for a session to appear (allows /frames to connect first).
    wait_deadline = time.time() + 5.0
    while CURRENT_SESSION is None and time.time() < wait_deadline:
        await asyncio.sleep(0.05)
    session = CURRENT_SESSION
    if session is None or session.frame_q is None:
        await ws.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    next_t = time.time()
    try:
        while True:
            if session.finished_event.is_set() and session.frame_q.empty():
                await ws.send_json({"event": "finished"})
                break
            try:
                jpeg = await asyncio.wait_for(session.frame_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            await ws.send_bytes(jpeg)
            next_t += period
            sleep_for = next_t - time.time()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                # Falling behind — reset pace anchor so we don't burst.
                next_t = time.time()
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ============================================================================
# CLI entrypoint
# ============================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive streaming SANA-WM demo (FastAPI + WebSocket).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--streaming_root", type=Path, default=DEFAULT_STREAMING_ROOT,
                   help="Directory holding sana_dit/, ltx2_causal_vae/, refiner_diffusers/, gemma3_12b/, and the yaml.")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--share", action="store_true",
                   help="Publish a public *.gradio.live URL via Gradio's share tunnel "
                        "(FRP). The local server still runs as usual; the tunnel just "
                        "forwards public traffic to it.")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Override the persistent disk cache root (default "
                        "$SANA_WM_STREAMING_CACHE_DIR or ~/.cache/sana_wm_streaming_demo).")
    p.add_argument("--no_warmup", action="store_true",
                   help="Skip the startup warmup session. First interactive session "
                        "will then pay the ~3 min cold torch.compile cost.")
    p.add_argument("--log_level", default="info")
    return p


def _start_share_tunnel(local_host: str, local_port: int, *, retries: int = 3) -> str:
    """Borrow Gradio's FRP-based share tunnel to publish a *.gradio.live URL.

    Connects directly to ``gradio-live.com:7000`` (the same FRP server
    Gradio's API auto-discovers) but **without TLS** on the internal
    frpc↔frps hop. Some clusters TLS-inspect outbound port 7000 and
    silently drop the handshake; the plain frp login works through the
    same middlebox. End-user traffic to ``*.gradio.live`` is still HTTPS —
    only the internal control channel is plain.

    Uses ``secrets.token_urlsafe(32)`` for the proxy name (same format
    Gradio uses internally). Set ``$GRADIO_SHARE_SERVER_ADDRESS`` to
    override the default endpoint.
    """
    import secrets
    from gradio import networking

    forward_host = "127.0.0.1" if local_host in ("0.0.0.0", "") else local_host
    server_addr = os.environ.get("GRADIO_SHARE_SERVER_ADDRESS", "gradio-live.com:7000")

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        token = secrets.token_urlsafe(32)
        try:
            # share_server_tls_certificate=None → frpc runs without --tls_enable.
            return networking.setup_tunnel(forward_host, local_port, token, server_addr, None)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            LOGGER.warning(
                f"[share] attempt {attempt}/{retries} failed: {exc!r}; "
                f"{'retrying…' if attempt < retries else 'giving up'}"
            )
            time.sleep(2.0)
    assert last_exc is not None
    raise last_exc


def main() -> None:
    args = _build_parser().parse_args()
    get_root_logger()
    logging.basicConfig(level=args.log_level.upper())
    app.state.streaming_root = args.streaming_root
    app.state.cache_dir = args.cache_dir
    app.state.do_warmup = not args.no_warmup

    if args.share:
        try:
            public_url = _start_share_tunnel(args.host, args.port)
            LOGGER.info(f"[share] public URL: {public_url}")
            print(f"\n[SANA-WM] Public URL: {public_url}\n", flush=True)
        except Exception as exc:
            LOGGER.exception("[share] failed to start tunnel; falling back to local-only")
            print(f"[SANA-WM] WARNING: share tunnel failed ({exc!r}); "
                  f"open http://{args.host}:{args.port}/ from this node.", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
