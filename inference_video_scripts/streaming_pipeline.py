"""Chunk-pipelined orchestrator for streaming Sana-WM inference.

Drives three CUDA streams (stage-1 DiT, LTX-2 refiner, causal LTX-2 VAE) so
one chunk is in flight per stage. Every AR chunk produces one decoded video
chunk, which is written progressively into an MP4 via :class:`StreamingMp4Writer`.

Cadence (canonical recipe ``distilled-4step + source-sink-1``):

* Stage-1 chunks of ``base_chunk_frames=3`` latents (chunk 0 absorbs the
  ``8k+1`` stride remainder and covers ``[0, 4)``).
* Refiner blocks of ``block_size=3`` latents; the sink at frame 0 is captured
  as the attention anchor but never refined.
* Decode chunks of ``block_size`` latents, plus the sink on chunk 0. The sink
  pixel frame is dropped on the way to ffmpeg.

The pipeline is 1:1 between stages: refiner block ``k`` depends on stage-1
chunk ``k``; decode chunk ``k`` depends on refiner block ``k``.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import time

import numpy as np
import torch

from diffusion.model.ltx2 import CausalVaeStreamingDecoder
from diffusion.refiner.diffusers_ltx2_refiner import RefinerChunkRunner
from inference_video_scripts.streaming_mp4_writer import StreamingMp4Writer


@dataclass
class StreamingPipelineConfig:
    """Static settings for one streaming-inference call."""

    sink_size: int = 1
    block_size: int = 3
    fps: int = 16
    output_path: str | Path = "streaming_output.mp4"
    mp4_crf: int = 18
    mp4_preset: str = "medium"
    drop_first_pixel: bool = True
    output_mode: str = "mp4"
    profile_cuda: bool = False
    sample_frames_path: str | Path | None = None
    sample_frame_stride: int = 0


@dataclass
class StreamingPipelineResult:
    output_path: Path | None
    n_pixel_frames: int
    n_refiner_blocks: int
    n_decode_chunks: int
    output_mode: str
    wall_seconds: float
    first_chunk_seconds: float | None
    first_chunk_frames: int | None
    steady_state_seconds: float | None
    steady_state_frames_per_second: float | None
    steady_state_realtime_factor: float | None
    frames_per_second: float
    realtime_factor: float
    stage1_cuda_seconds: float | None = None
    refiner_cuda_seconds: float | None = None
    decode_cuda_seconds: float | None = None
    sample_frames_path: Path | None = None
    sampled_frame_count: int = 0
    sampled_frame_indices: list[int] | None = None


@torch.inference_mode()
def run_streaming_inference(
    *,
    stage1_chunk_iter: Iterator[tuple[int, torch.Tensor, int, int]],
    n_stage1_chunks: int,
    z_init: torch.Tensor,
    refiner_runner: RefinerChunkRunner,
    vae_streaming_decoder: CausalVaeStreamingDecoder,
    pixel_h: int,
    pixel_w: int,
    config: StreamingPipelineConfig,
    logger=None,
) -> StreamingPipelineResult:
    """Drive the three-stage chunked pipeline end-to-end.

    Args:
        stage1_chunk_iter: Iterator (typically from
            ``SelfForcingFlowEulerCamCtrl.sample_chunks``) yielding
            ``(chunk_idx, latent_view, start_f, end_f)`` after each AR chunk.
            ``latent_view`` is a view into the sampler's in-place latents
            tensor; the orchestrator defensively mirror-copies it so the
            refiner never races with subsequent stage-1 writes.
        n_stage1_chunks: Number of chunks the iterator will yield.
        z_init: ``(B, C, T_latent, H_lat, W_lat)`` initial latent tensor with
            sink populated at ``[:, :, :sink_size]``.
        refiner_runner: A :class:`RefinerChunkRunner` already configured with
            the prompt, sigmas, fps and seed.
        vae_streaming_decoder: A :class:`CausalVaeStreamingDecoder` wrapping
            ``AutoencoderKLCausalLTX2Video``; reset before the first call.
        pixel_h, pixel_w: Decoded frame dimensions (``vae_stride * H_lat``).
        config: Static configuration.
        logger: Optional logger; falls back to ``print``.

    Returns:
        :class:`StreamingPipelineResult` describing the produced MP4.
    """
    log = (logger.info if logger is not None else print)

    sink_size = int(config.sink_size)
    block_size = int(config.block_size)
    T_latent = int(z_init.shape[2])

    n_active = max(T_latent - sink_size, 0)
    n_refiner = n_active // block_size
    if n_refiner * block_size != n_active:
        raise ValueError(
            f"Active latent frames ({n_active}) must be divisible by "
            f"block_size ({block_size}); got remainder {n_active % block_size}."
        )
    if n_stage1_chunks <= 0:
        raise ValueError("n_stage1_chunks must be > 0.")
    n_decode = n_refiner
    output_mode = str(config.output_mode)
    if output_mode not in {"mp4", "cpu", "discard"}:
        raise ValueError(f"output_mode must be one of mp4/cpu/discard; got {output_mode!r}.")

    log(
        f"[stream] T_latent={T_latent} sink={sink_size} block={block_size} "
        f"stage1_chunks={n_stage1_chunks} refiner_blocks={n_refiner} "
        f"decode_chunks={n_decode} output_mode={output_mode}"
    )

    # Pre-allocated mirror buffers. Slices into these are handed across
    # streams; the storage is long-lived so no record_stream is needed.
    latents_full = torch.empty_like(z_init)
    latents_full[:, :, :sink_size] = z_init[:, :, :sink_size]
    refined_full = torch.empty_like(z_init)
    refined_full[:, :, :sink_size] = z_init[:, :, :sink_size]

    device = z_init.device
    stage1_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    refiner_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    decode_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def _on(stream):
        return torch.cuda.stream(stream) if stream is not None else nullcontext()

    def _new_event():
        return torch.cuda.Event() if device.type == "cuda" else None

    def _new_timing_event():
        if device.type != "cuda" or not bool(config.profile_cuda):
            return None
        return torch.cuda.Event(enable_timing=True)

    def _record(event, stream):
        if event is not None and stream is not None:
            event.record(stream)

    def _wait(stream, event):
        if stream is not None and event is not None:
            stream.wait_event(event)

    stage1_events = [_new_event() for _ in range(n_stage1_chunks)]
    refiner_events = [_new_event() for _ in range(n_refiner)]
    decode_events = [_new_event() for _ in range(n_decode)]
    stage1_timing_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    refiner_timing_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    decode_timing_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    vae_streaming_decoder.reset()

    pending: deque[tuple[torch.cuda.Event | None, torch.Tensor | int, int]] = deque()
    writer = None
    if output_mode == "mp4":
        writer = StreamingMp4Writer(
            config.output_path,
            height=int(pixel_h),
            width=int(pixel_w),
            fps=int(config.fps),
            crf=int(config.mp4_crf),
            preset=str(config.mp4_preset),
        )

    n_pixel_frames = 0
    sample_frames: list[np.ndarray] = []
    sample_frame_indices: list[int] = []
    sample_frames_path = Path(config.sample_frames_path) if config.sample_frames_path is not None else None
    sample_frame_stride = int(config.sample_frame_stride)
    if sample_frames_path is not None and sample_frame_stride <= 0:
        raise ValueError("sample_frame_stride must be > 0 when sample_frames_path is set.")

    def _collect_sample_frames(pixel_np: np.ndarray, frame_base: int) -> None:
        if sample_frames_path is None:
            return
        for local_idx in range(0, int(pixel_np.shape[0])):
            frame_idx = int(frame_base + local_idx)
            if frame_idx % sample_frame_stride == 0:
                sample_frames.append(pixel_np[local_idx].copy())
                sample_frame_indices.append(frame_idx)

    t_start = time.perf_counter()
    first_chunk_seconds: float | None = None
    first_chunk_frames: int | None = None
    try:
        # Schedule per timestep:
        #   t=0:  stage1[0]
        #   t=1:  stage1[1], refiner[0]
        #   t=2:  stage1[2], refiner[1], decode[0]
        #   ...
        n_iters = max(n_stage1_chunks, n_refiner + 1, n_decode + 2)

        for t in range(n_iters):
            # --- stage-1 chunk t ---
            if t < n_stage1_chunks:
                with _on(stage1_stream):
                    timing_start = _new_timing_event()
                    timing_end = _new_timing_event()
                    _record(timing_start, stage1_stream)
                    _, latent_view, start_f, end_f = next(stage1_chunk_iter)
                    latents_full[:, :, start_f:end_f].copy_(latent_view, non_blocking=True)
                    _record(timing_end, stage1_stream)
                    if timing_start is not None and timing_end is not None:
                        stage1_timing_events.append((timing_start, timing_end))
                _record(stage1_events[t], stage1_stream)

            # --- refiner block t - 1 ---
            k_ref = t - 1
            if 0 <= k_ref < n_refiner:
                _wait(refiner_stream, stage1_events[k_ref])
                block_start = sink_size + k_ref * block_size
                block_end = block_start + block_size
                with _on(refiner_stream):
                    timing_start = _new_timing_event()
                    timing_end = _new_timing_event()
                    _record(timing_start, refiner_stream)
                    clean_block = latents_full[:, :, block_start:block_end]
                    sink_seed = latents_full[:, :, :sink_size] if k_ref == 0 else None
                    refined_block = refiner_runner.refine_block(
                        block_idx=k_ref,
                        clean_block=clean_block,
                        block_start=block_start,
                        block_end=block_end,
                        sink_seed_frames=sink_seed,
                    )
                    refined_full[:, :, block_start:block_end].copy_(refined_block, non_blocking=True)
                    _record(timing_end, refiner_stream)
                    if timing_start is not None and timing_end is not None:
                        refiner_timing_events.append((timing_start, timing_end))
                _record(refiner_events[k_ref], refiner_stream)

            # --- decode chunk t - 2 ---
            k_dec = t - 2
            if 0 <= k_dec < n_decode:
                _wait(decode_stream, refiner_events[k_dec])
                if k_dec == 0:
                    z_slice = refined_full[:, :, : sink_size + block_size]
                else:
                    z_slice = refined_full[
                        :, :, sink_size + k_dec * block_size : sink_size + (k_dec + 1) * block_size
                    ]
                with _on(decode_stream):
                    timing_start = _new_timing_event()
                    timing_end = _new_timing_event()
                    _record(timing_start, decode_stream)
                    pixel_chunk = vae_streaming_decoder.decode_chunk(z_slice)
                    if output_mode == "discard":
                        n_frames = int(pixel_chunk.shape[2])
                        if k_dec == 0 and config.drop_first_pixel:
                            n_frames -= 1
                        pixel_out = max(0, n_frames)
                    else:
                        pixel_uint8 = (
                            (pixel_chunk.float() * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
                        )
                        pixel_hwc = pixel_uint8.permute(0, 2, 3, 4, 1).contiguous()
                        pixel_out = pixel_hwc.to("cpu", non_blocking=True)
                    _record(timing_end, decode_stream)
                    if timing_start is not None and timing_end is not None:
                        decode_timing_events.append((timing_start, timing_end))
                _record(decode_events[k_dec], decode_stream)
                pending.append((decode_events[k_dec], pixel_out, k_dec))

            # --- Flush any ready decoded chunks. In discard mode this only
            # retires the CUDA work; no CPU copy or encoder path is exercised.
            while pending and (pending[0][0] is None or pending[0][0].query()):
                _event, _pixel_out, _k = pending.popleft()
                if first_chunk_seconds is None:
                    first_chunk_seconds = time.perf_counter() - t_start
                if output_mode == "discard":
                    n_frames = int(_pixel_out)
                else:
                    pixel_np = _pixel_out.numpy()[0]
                    if _k == 0 and config.drop_first_pixel:
                        pixel_np = pixel_np[1:]
                    n_frames = int(pixel_np.shape[0])
                    _collect_sample_frames(pixel_np, n_pixel_frames)
                    if output_mode == "mp4":
                        assert writer is not None
                        writer.write_chunk(pixel_np)
                n_pixel_frames += n_frames
                if first_chunk_frames is None:
                    first_chunk_frames = n_frames

        # Drain.
        while pending:
            _event, _pixel_out, _k = pending.popleft()
            if _event is not None:
                _event.synchronize()
            if first_chunk_seconds is None:
                first_chunk_seconds = time.perf_counter() - t_start
            if output_mode == "discard":
                n_frames = int(_pixel_out)
            else:
                pixel_np = _pixel_out.numpy()[0]
                if _k == 0 and config.drop_first_pixel:
                    pixel_np = pixel_np[1:]
                n_frames = int(pixel_np.shape[0])
                _collect_sample_frames(pixel_np, n_pixel_frames)
                if output_mode == "mp4":
                    assert writer is not None
                    writer.write_chunk(pixel_np)
            n_pixel_frames += n_frames
            if first_chunk_frames is None:
                first_chunk_frames = n_frames

        out_path = writer.close() if writer is not None else None
        if sample_frames_path is not None:
            sample_frames_path.parent.mkdir(parents=True, exist_ok=True)
            frames = (
                np.stack(sample_frames, axis=0)
                if sample_frames
                else np.empty((0, int(pixel_h), int(pixel_w), 3), dtype=np.uint8)
            )
            np.savez_compressed(
                sample_frames_path,
                frames=frames,
                frame_indices=np.asarray(sample_frame_indices, dtype=np.int64),
            )
    except Exception:
        if writer is not None:
            writer.close()
        raise

    wall_seconds = time.perf_counter() - t_start
    frames_per_second = float(n_pixel_frames) / wall_seconds if wall_seconds > 0.0 else 0.0
    realtime_factor = frames_per_second / float(config.fps) if config.fps else 0.0
    steady_state_seconds: float | None = None
    steady_state_frames_per_second: float | None = None
    steady_state_realtime_factor: float | None = None
    if first_chunk_seconds is not None and first_chunk_frames is not None:
        steady_frames = max(0, int(n_pixel_frames) - int(first_chunk_frames))
        steady_seconds = wall_seconds - float(first_chunk_seconds)
        if steady_frames > 0 and steady_seconds > 0.0:
            steady_state_seconds = steady_seconds
            steady_state_frames_per_second = float(steady_frames) / steady_seconds
            steady_state_realtime_factor = steady_state_frames_per_second / float(config.fps) if config.fps else 0.0

    def _sum_cuda_seconds(events: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> float | None:
        if device.type != "cuda" or not events:
            return None
        total_ms = 0.0
        for start, end in events:
            total_ms += float(start.elapsed_time(end))
        return total_ms / 1000.0

    stage1_cuda_seconds = _sum_cuda_seconds(stage1_timing_events)
    refiner_cuda_seconds = _sum_cuda_seconds(refiner_timing_events)
    decode_cuda_seconds = _sum_cuda_seconds(decode_timing_events)
    log(
        f"[stream] output_mode={output_mode} frames={n_pixel_frames} "
        f"wall={wall_seconds:.3f}s fps={frames_per_second:.3f} "
        f"realtime={realtime_factor:.3f}x first_chunk={first_chunk_seconds} "
        f"steady_fps={steady_state_frames_per_second} steady_realtime={steady_state_realtime_factor} "
        f"cuda_stage1={stage1_cuda_seconds} cuda_refiner={refiner_cuda_seconds} "
        f"cuda_decode={decode_cuda_seconds} "
        f"sample_frames={sample_frames_path} sample_count={len(sample_frame_indices)} "
        f"path={out_path} (refiner_blocks={n_refiner}, decode_chunks={n_decode})"
    )
    return StreamingPipelineResult(
        output_path=out_path,
        n_pixel_frames=n_pixel_frames,
        n_refiner_blocks=n_refiner,
        n_decode_chunks=n_decode,
        output_mode=output_mode,
        wall_seconds=wall_seconds,
        first_chunk_seconds=first_chunk_seconds,
        first_chunk_frames=first_chunk_frames,
        steady_state_seconds=steady_state_seconds,
        steady_state_frames_per_second=steady_state_frames_per_second,
        steady_state_realtime_factor=steady_state_realtime_factor,
        frames_per_second=frames_per_second,
        realtime_factor=realtime_factor,
        stage1_cuda_seconds=stage1_cuda_seconds,
        refiner_cuda_seconds=refiner_cuda_seconds,
        decode_cuda_seconds=decode_cuda_seconds,
        sample_frames_path=sample_frames_path,
        sampled_frame_count=len(sample_frame_indices),
        sampled_frame_indices=sample_frame_indices,
    )
