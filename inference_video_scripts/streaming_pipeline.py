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

    log(
        f"[stream] T_latent={T_latent} sink={sink_size} block={block_size} "
        f"stage1_chunks={n_stage1_chunks} refiner_blocks={n_refiner} decode_chunks={n_decode}"
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

    def _record(event, stream):
        if event is not None and stream is not None:
            event.record(stream)

    def _wait(stream, event):
        if stream is not None and event is not None:
            stream.wait_event(event)

    stage1_events = [_new_event() for _ in range(n_stage1_chunks)]
    refiner_events = [_new_event() for _ in range(n_refiner)]
    decode_events = [_new_event() for _ in range(n_decode)]

    vae_streaming_decoder.reset()

    # Pixel chunks awaiting CPU-side ffmpeg writes.
    pending: deque[tuple[torch.cuda.Event | None, torch.Tensor, int]] = deque()

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
                    _, latent_view, start_f, end_f = next(stage1_chunk_iter)
                    latents_full[:, :, start_f:end_f].copy_(latent_view, non_blocking=True)
                _record(stage1_events[t], stage1_stream)

            # --- refiner block t - 1 ---
            k_ref = t - 1
            if 0 <= k_ref < n_refiner:
                _wait(refiner_stream, stage1_events[k_ref])
                block_start = sink_size + k_ref * block_size
                block_end = block_start + block_size
                with _on(refiner_stream):
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
                    pixel_chunk = vae_streaming_decoder.decode_chunk(z_slice)
                    pixel_uint8 = (
                        (pixel_chunk.float() * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
                    )
                    pixel_hwc = pixel_uint8.permute(0, 2, 3, 4, 1).contiguous()
                    pixel_cpu = pixel_hwc.to("cpu", non_blocking=True)
                _record(decode_events[k_dec], decode_stream)
                pending.append((decode_events[k_dec], pixel_cpu, k_dec))

            # --- Flush any ready decoded chunks to ffmpeg ---
            while pending and (pending[0][0] is None or pending[0][0].query()):
                _event, _pixel_cpu, _k = pending.popleft()
                pixel_np = _pixel_cpu.numpy()[0]
                if _k == 0 and config.drop_first_pixel:
                    pixel_np = pixel_np[1:]
                writer.write_chunk(pixel_np)
                n_pixel_frames += int(pixel_np.shape[0])

        # Drain.
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
