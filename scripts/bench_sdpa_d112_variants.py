#!/usr/bin/env python3
"""Microbenchmark Sana-WM D=112 SDPA padding versus direct dispatch.

The Stage-1 softmax attention path historically pads head_dim 112 to 128 before
calling PyTorch SDPA. On Blackwell/cuDNN, direct D=112 SDPA may be legal and can
avoid three pads plus a slice. This script measures the exact hot shapes before
using ``SANA_WM_SDPA_D112_DIRECT=1`` in the full pipeline.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F


def _parse_case(text: str) -> tuple[str, int, int]:
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("case must be name:q_len:kv_len")
    name, q_len, kv_len = parts
    return name, int(q_len), int(kv_len)


def _sync() -> None:
    torch.cuda.synchronize()


def _sdpa_pad(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_pad = F.pad(q, (0, 16))
    k_pad = F.pad(k, (0, 16))
    v_pad = F.pad(v, (0, 16))
    return F.scaled_dot_product_attention(q_pad, k_pad, v_pad)[..., : q.shape[-1]]


def _sdpa_direct(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v)


def _bench(fn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        out = fn(q, k, v)
    _sync()

    times: list[float] = []
    peak = 0.0
    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats()
        _sync()
        t0 = time.perf_counter()
        out = fn(q, k, v)
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)
        peak = max(peak, torch.cuda.max_memory_allocated() / 1024**3)
        del out
    return statistics.median(times), peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--dim", type=int, default=112)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        default=[
            ("chunk0_193", 3520, 3520),
            ("chunk1_193", 2640, 6160),
            ("chunk7_193", 2640, 22000),
        ],
        help="Benchmark case as name:q_len:kv_len. Can be repeated.",
    )
    parser.add_argument("--include-full-961", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(True)

    cases = list(args.case)
    if args.include_full_961:
        cases.append(("full_961", 106480, 106480))

    print(
        "device,batch,heads,dim,case,q_len,kv_len,pad_ms,direct_ms,speedup,direct_peak_gib,pad_peak_gib"
    )
    for name, q_len, kv_len in cases:
        q = torch.randn(args.batch, args.heads, q_len, args.dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(args.batch, args.heads, kv_len, args.dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(args.batch, args.heads, kv_len, args.dim, device="cuda", dtype=torch.bfloat16)
        _sync()

        pad_ms, pad_peak = _bench(_sdpa_pad, q, k, v, args.warmup, args.iters)
        direct_ms, direct_peak = _bench(_sdpa_direct, q, k, v, args.warmup, args.iters)
        speedup = pad_ms / direct_ms if direct_ms > 0 else float("nan")
        print(
            f"{torch.cuda.get_device_name()},{args.batch},{args.heads},{args.dim},"
            f"{name},{q_len},{kv_len},{pad_ms:.3f},{direct_ms:.3f},"
            f"{speedup:.4f},{direct_peak:.3f},{pad_peak:.3f}",
            flush=True,
        )
        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
