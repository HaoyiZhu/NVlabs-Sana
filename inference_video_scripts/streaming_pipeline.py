"""Chunk-pipelined inference orchestrator for Sana-WM streaming generation.

Drives three CUDA streams on a single GPU so stage-1 (Sana DiT), the LTX-2 AR
refiner, and the causal LTX-2 VAE decoder overlap one chunk in flight per stage.
The pipeline emits one decoded video chunk per AR block straight into a
progressive MP4 file via :class:`StreamingMp4Writer`.

Chunk cadence (canonical recipe ``distilled-3step + source-sink-1``):

  - ``base_chunk_frames`` (=3 latents) stage-1 chunks  -> 41 for 121 latents
  - ``block_size`` (=3 latents) refiner blocks         -> 40 (sink not refined)
  - causal-VAE decode chunks                            -> 40

  Refiner block ``k`` covers latent frames ``[1 + 3k, 1 + 3k + 3)``; the sink
  latent at frame 0 is pre-captured as the attention anchor but never refined.

  Decode chunk 0 takes latents ``[0, 4)`` (sink + refiner block 0); subsequent
  decode chunks take ``[1 + 3k, 1 + 3k + 3)``. The very first decoded pixel
  frame (corresponding to the sink) is dropped on the way to ffmpeg so the
  output starts from the first refined frame, mirroring the legacy
  ``_refine`` contract.

Dependency / event graph::

  stage1[k] --(stage1_event[k])--> refiner[k] --(refiner_event[k])--> decode[k]

  Stage-1's ``create_autoregressive_segments`` absorbs the temporal-stride
  remainder into chunk 0, so for ``base_chunk_frames=3``, stage-1 chunk 0
  covers latent frames ``[0, 4)`` and subsequent chunks cover ``[4+3k, 4+3k+3)``.
  Refiner block ``k`` needs frames ``[3k+1, 3k+4)``, which lies entirely
  inside stage-1 chunk ``k`` (k=0: ``[1,4)`` inside ``[0,4)``; k>=1: identical
  ranges). The dependency is therefore 1:1 between stage-1 chunks and
  refiner blocks. n_stage1 == n_refiner == n_decode for the canonical recipe.

The orchestrator is intentionally a single Python loop so kernel ordering on
each stream is FIFO, eliminating reentrancy hazards on the mutable cache state
held by both the refiner's ``RefinerChunkRunner`` and the VAE's
``_decoder_cache``. Cross-stream handoffs use CUDA events; tensors handed
across streams are slices of long-lived pre-allocated buffers so we never
need ``record_stream``.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

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
    """Latent frames per refiner block (==base_chunk_frames in stage-1)."""

    fps: int = 16
    output_path: str | Path = "streaming_output.mp4"
    mp4_crf: int = 18
    mp4_preset: str = "medium"

    drop_first_pixel: bool = True
    """Drop pixel frame 0 (sink) from the first decoded chunk. Set False to
    keep the sink for debugging visual alignment."""


@dataclass
class StreamingPipelineResult:
    output_path: Path
    n_pixel_frames: int
    n_refiner_blocks: int
    n_decode_chunks: int


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
            ``SelfForcingFlowEulerCamCtrl.sample_chunks``) yielding tuples
            ``(chunk_idx, latent_view, start_f, end_f)`` after each AR chunk
            completes. ``latent_view`` may be a view into the sampler's
            in-place working tensor; the orchestrator copies it into its own
            buffer so subsequent stage-1 chunks don't race against the refiner.
        n_stage1_chunks: Number of chunks the iterator will yield. Used to
            size the pipeline and pre-allocate event arrays.
        z_init: ``(B, C, T_latent, H_lat, W_lat)`` initial latent tensor with
            sink frame already populated at ``[:, :, :sink_size]``. The
            orchestrator clones this into its working buffer.
        refiner_runner: A :class:`RefinerChunkRunner` already instantiated
            with the prompt, sigmas, fps and seed.
        vae_streaming_decoder: A :class:`CausalVaeStreamingDecoder` wrapping
            an ``AutoencoderKLCausalLTX2Video``; will be reset before
            the first call.
        pixel_h: Pixel height of decoded frames (vae_stride * H_lat).
        pixel_w: Pixel width of decoded frames (vae_stride * W_lat).
        config: Static configuration (sink size, block size, fps, output mp4).
        logger: Optional logger; if None, prints to stdout.

    Returns:
        :class:`StreamingPipelineResult` describing the produced video.
    """
    log = (logger.info if logger is not None else print)

    sink_size = int(config.sink_size)
    block_size = int(config.block_size)
    T_latent = int(z_init.shape[2])
    B, C, _, H_lat, W_lat = z_init.shape

    n_active = max(T_latent - sink_size, 0)
    n_refiner = n_active // block_size
    if n_refiner * block_size != n_active:
        raise ValueError(
            f"Active latent frames ({n_active}) must be divisible by "
            f"block_size ({block_size}); got {n_active} % {block_size} != 0."
        )
    if n_stage1_chunks <= 0:
        raise ValueError("n_stage1_chunks must be > 0.")
    # decode chunk 0 consumes sink + refiner block 0; subsequent decode chunks
    # consume one refiner block each.
    n_decode = n_refiner

    log(
        f"[stream] T_latent={T_latent} sink={sink_size} block={block_size} "
        f"stage1_chunks={n_stage1_chunks} refiner_blocks={n_refiner} decode_chunks={n_decode}"
    )

    device = z_init.device
    dtype = z_init.dtype

    # Pre-allocated buffers (stage-1 mirror + refined output). Slices into
    # these are handed across streams; the storage is long-lived so no
    # record_stream is needed.
    latents_full = torch.empty_like(z_init)
    latents_full[:, :, :sink_size] = z_init[:, :, :sink_size]
    refined_full = torch.empty_like(z_init)
    refined_full[:, :, :sink_size] = z_init[:, :, :sink_size]

    stage1_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    refiner_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    decode_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def _on(stream):
        return torch.cuda.stream(stream) if stream is not None else nullcontext()

    def _new_event():
        return torch.cuda.Event() if device.type == "cuda" else None

    def _record(event, stream):
        if event is not None and stream is not None:
            event.record(stream)

    def _wait(stream, event):
        if stream is not None and event is not None:
            stream.wait_event(event)

    stage1_events: list[torch.cuda.Event | None] = [_new_event() for _ in range(n_stage1_chunks)]
    refiner_events: list[torch.cuda.Event | None] = [_new_event() for _ in range(n_refiner)]
    decode_events: list[torch.cuda.Event | None] = [_new_event() for _ in range(n_decode)]

    vae_streaming_decoder.reset()

    # Pixel chunks pending CPU-side ffmpeg writes.
    pending: deque[tuple[torch.cuda.Event | None, torch.Tensor, int]] = deque()

    # Open progressive MP4 writer.
    writer = StreamingMp4Writer(
        config.output_path,
        height=int(pixel_h),
        width=int(pixel_w),
        fps=int(config.fps),
        crf=int(config.mp4_crf),
        preset=str(config.mp4_preset),
    )

    n_pixel_frames = 0
    try:
        # The schedule advances all three stages per timestep:
        #   t=0:  stage1[0]
        #   t=1:  stage1[1], refiner[0]            (needs stage1_event[0])
        #   t=2:  stage1[2], refiner[1], decode[0] (needs refiner_event[0])
        #   ...
        # Total iterations = max(n_stage1, n_refiner+1, n_decode+2).
        n_iters = max(n_stage1_chunks, n_refiner + 1, n_decode + 2)

        for t in range(n_iters):
            # --- stage-1 chunk t ---
            if t < n_stage1_chunks:
                with _on(stage1_stream):
                    chunk_idx, latent_view, start_f, end_f = next(stage1_chunk_iter)
                    # Defensive copy into our buffer so subsequent stage-1
                    # writes never alias what refiner is reading.
                    latents_full[:, :, start_f:end_f].copy_(latent_view, non_blocking=True)
                _record(stage1_events[t], stage1_stream)

            # --- refiner block k_ref = t - 1 ---
            k_ref = t - 1
            if 0 <= k_ref < n_refiner:
                # Refiner block k_ref needs latent frames [3*k_ref + 1, 3*k_ref + 4).
                # Stage-1 chunk 0 covers [0, 4) (remainder absorbed), so refiner[0]
                # depends on stage1[0]. For k_ref >= 1, stage-1 chunk k_ref covers
                # exactly [3*k_ref + 1, 3*k_ref + 4), so refiner[k_ref] depends on
                # stage1[k_ref]. Either way the dependency is 1:1.
                _wait(refiner_stream, stage1_events[k_ref])
                block_start = sink_size + k_ref * block_size
                block_end = block_start + block_size
                with _on(refiner_stream):
                    clean_block = latents_full[:, :, block_start:block_end]
                    sink_seed = (
                        latents_full[:, :, :sink_size] if k_ref == 0 else None
                    )
                    refined_block = refiner_runner.refine_block(
                        block_idx=k_ref,
                        clean_block=clean_block,
                        block_start=block_start,
                        block_end=block_end,
                        sink_seed_frames=sink_seed,
                    )
                    refined_full[:, :, block_start:block_end].copy_(refined_block, non_blocking=True)
                _record(refiner_events[k_ref], refiner_stream)

            # --- decode chunk k_dec = t - 2 ---
            k_dec = t - 2
            if 0 <= k_dec < n_decode:
                _wait(decode_stream, refiner_events[k_dec])
                if k_dec == 0:
                    z_slice = refined_full[:, :, : sink_size + block_size]
                else:
                    z_slice = refined_full[
                        :,
                        :,
                        sink_size + k_dec * block_size : sink_size + (k_dec + 1) * block_size,
                    ]
                with _on(decode_stream):
                    pixel_chunk = vae_streaming_decoder.decode_chunk(z_slice)
                    # (B, 3, T_pix, H, W) in [-1, 1]; convert to (T_pix, H, W, 3) uint8.
                    pixel_uint8 = (
                        (pixel_chunk.float() * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
                    )
                    pixel_hwc = pixel_uint8.permute(0, 2, 3, 4, 1).contiguous()
                    pixel_cpu = pixel_hwc.to("cpu", non_blocking=True)
                _record(decode_events[k_dec], decode_stream)
                pending.append((decode_events[k_dec], pixel_cpu, k_dec))

            # --- Flush any ready decoded chunks to ffmpeg ---
            while pending and (
                pending[0][0] is None or pending[0][0].query()
            ):
                _event, _pixel_cpu, _k = pending.popleft()
                pixel_np = _pixel_cpu.numpy()[0]
                if _k == 0 and config.drop_first_pixel:
                    pixel_np = pixel_np[1:]
                writer.write_chunk(pixel_np)
                n_pixel_frames += int(pixel_np.shape[0])

        # Drain any remaining writes.
        while pending:
            _event, _pixel_cpu, _k = pending.popleft()
            if _event is not None:
                _event.synchronize()
            pixel_np = _pixel_cpu.numpy()[0]
            if _k == 0 and config.drop_first_pixel:
                pixel_np = pixel_np[1:]
            writer.write_chunk(pixel_np)
            n_pixel_frames += int(pixel_np.shape[0])

        out_path = writer.close()
    except Exception:
        # Make sure the ffmpeg subprocess doesn't survive on error.
        writer.close()
        raise

    log(
        f"[stream] wrote {n_pixel_frames} pixel frames to {out_path} "
        f"(refiner_blocks={n_refiner}, decode_chunks={n_decode})"
    )
    return StreamingPipelineResult(
        output_path=out_path,
        n_pixel_frames=n_pixel_frames,
        n_refiner_blocks=n_refiner,
        n_decode_chunks=n_decode,
    )
