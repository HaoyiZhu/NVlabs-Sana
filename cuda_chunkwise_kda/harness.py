#!/usr/bin/env python3
"""KDA harness: Triton baseline + CUDA candidate for chunkwise BiGDN forward.

Correctness oracle = the Triton `fused_bigdn_bidi_chunkwise` on identical inputs.
Timing = CUDA-event median over `--iters` after `--warmup`.

Usage:
  python cuda_chunkwise_kda/harness.py                 # Triton baseline + per-phase
  python cuda_chunkwise_kda/harness.py --impl cuda --check
"""
from __future__ import annotations

import argparse
import os
import sys
import json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from diffusion.model.ops.fused_gdn_chunkwise import (
    fused_bigdn_bidi_chunkwise,
    phase_a,
    phase_b_triton,
    phase_c,
)


def make_inputs(B, F, S, H, D, device, dtype, seed=0, scale=0.1, beta_scale=0.02):
    """Valid, finite production-shape inputs. Magnitudes tuned so the GDN
    recurrence stays O(1) over F frames (near-identity I-P_kv, decay<1)."""
    g = torch.Generator(device=device).manual_seed(seed)
    N = F * S
    BH = B * H

    def rn(*shape, s=1.0):
        return torch.randn(*shape, device=device, generator=g) * s

    qkv = (rn(B, N, 3, H, D, s=scale)).to(dtype)
    # 1/rms — strictly positive, ~1
    q_inv_rms = (0.75 + 0.5 * torch.rand(B, N, device=device, generator=g)).float()
    k_inv_rms = (0.75 + 0.5 * torch.rand(B, N, device=device, generator=g)).float()
    q_norm_w = (1.0 + 0.1 * rn(H * D)).float()
    k_norm_w = (1.0 + 0.1 * rn(H * D)).float()
    theta = torch.rand(N, D, device=device, generator=g) * 6.2831853
    rope_cos = theta.cos().float().contiguous()
    rope_sin = theta.sin().float().contiguous()
    beta = (beta_scale * torch.rand(BH, N, device=device, generator=g)).float()
    decay = (0.90 + 0.09 * torch.rand(BH, F, device=device, generator=g)).float()
    return dict(
        qkv=qkv, q_inv_rms=q_inv_rms, k_inv_rms=k_inv_rms,
        q_norm_w=q_norm_w, k_norm_w=k_norm_w, rope_cos=rope_cos, rope_sin=rope_sin,
        beta=beta, decay=decay, F=F, S=S,
    )


def run_triton(inp, dot_precision=0):
    return fused_bigdn_bidi_chunkwise(
        inp["qkv"], inp["q_inv_rms"], inp["k_inv_rms"], inp["q_norm_w"],
        inp["k_norm_w"], inp["rope_cos"], inp["rope_sin"], inp["beta"],
        inp["decay"], F=inp["F"], S=inp["S"], dot_precision=dot_precision,
    )


def time_ms(fn, warmup, iters):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return t[len(t) // 2]


def per_phase_times(inp, warmup, iters, dot_precision=0):
    qkv = inp["qkv"]
    F, S = inp["F"], inp["S"]

    def a():
        return phase_a(qkv, inp["beta"], inp["q_inv_rms"], inp["k_inv_rms"],
                       inp["q_norm_w"], inp["k_norm_w"], inp["rope_cos"],
                       inp["rope_sin"], F=F, S=S, dot_precision=dot_precision)

    I_P_kv, A, I_P_z, B_z = a()

    def b():
        return phase_b_triton(I_P_kv, A, I_P_z, B_z, inp["decay"], F=F,
                              dot_precision=dot_precision, direction=0,
                              combined_history=True)

    M_hist, z_hist, _, _ = b()

    def c():
        return phase_c(qkv, inp["q_inv_rms"], inp["q_norm_w"], inp["rope_cos"],
                       inp["rope_sin"], M_hist, z_hist, F=F, S=S,
                       dot_precision=dot_precision)

    ta = time_ms(a, warmup, iters)
    tb = time_ms(b, warmup, iters)
    tc = time_ms(c, warmup, iters)
    return ta, tb, tc


def rel_err(out, ref):
    out = out.float()
    ref = ref.float()
    denom = ref.abs().clamp_min(1e-4)
    re = (out - ref).abs() / denom
    return re.max().item(), re.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--F", type=int, default=11)
    ap.add_argument("--S", type=int, default=920)
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--impl", choices=["triton", "cuda"], default="triton")
    ap.add_argument("--cmode", choices=["c", "ac"], default="c",
                    help="c=Triton A + CUDA C-fused (best); ac=full CUDA A+C")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "benchmark.csv"))
    args = ap.parse_args()

    dev = torch.device("cuda")
    dtype = torch.bfloat16
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"# device={name} sm={cap[0]}{cap[1]} torch={torch.__version__}")
    print(f"# B={args.B} F={args.F} S={args.S} H={args.H} D={args.D} "
          f"N={args.F*args.S} BH={args.B*args.H} dtype={dtype}")

    inp = make_inputs(args.B, args.F, args.S, args.H, args.D, dev, dtype, args.seed)

    ref = run_triton(inp)
    assert torch.isfinite(ref).all(), "Triton ref produced NaN/Inf — reduce input scale"
    print(f"# triton out: shape={tuple(ref.shape)} dtype={ref.dtype} "
          f"abs_mean={ref.float().abs().mean():.4e} abs_max={ref.float().abs().max():.4e}")

    t_triton = time_ms(lambda: run_triton(inp), args.warmup, args.iters)
    ta, tb, tc = per_phase_times(inp, args.warmup, args.iters)
    print(f"\n# TRITON  end2end={t_triton:.4f} ms   "
          f"[phaseA={ta:.4f}  phaseB={tb:.4f}  phaseC={tc:.4f}  sum={ta+tb+tc:.4f}]")

    row = dict(impl="triton", device=name, sm=f"{cap[0]}{cap[1]}", B=args.B, F=args.F,
               S=args.S, H=args.H, D=args.D, ms=round(t_triton, 4),
               pa=round(ta, 4), pb=round(tb, 4), pc=round(tc, 4))

    if args.impl == "cuda":
        from cuda_impl import run_cuda, build, cuda_phase_c_fused  # noqa
        from diffusion.model.ops.fused_gdn_chunkwise import phase_a, phase_b_triton
        build()
        out = run_cuda(inp, mode=args.cmode)
        assert torch.isfinite(out).all(), "CUDA out has NaN/Inf"
        mx, mn = rel_err(out, ref)
        ok = mx <= 2e-2 and mn <= 2e-3
        print(f"\n# CUDA check: max_rel={mx:.3e} mean_rel={mn:.3e} -> "
              f"{'PASS' if ok else 'FAIL'}")
        if args.check and not ok:
            print("CORRECTNESS FAIL")
            sys.exit(1)
        # isolate CUDA Phase C-fused (Triton A/B already counted above)
        F, S = inp["F"], inp["S"]
        Ip, A, Iz, Bz = phase_a(inp["qkv"], inp["beta"], inp["q_inv_rms"], inp["k_inv_rms"],
                                inp["q_norm_w"], inp["k_norm_w"], inp["rope_cos"], inp["rope_sin"], F=F, S=S)
        Mh, zh, _, _ = phase_b_triton(Ip, A, Iz, Bz, inp["decay"], F=F, direction=0, combined_history=True)
        t_cfused = time_ms(lambda: cuda_phase_c_fused(inp, Mh, zh), args.warmup, args.iters)
        t_cuda = time_ms(lambda: run_cuda(inp, mode=args.cmode), args.warmup, args.iters)
        print(f"# CUDA    Cfused={t_cfused:.4f} ms (vs triton C+divide={tc:.4f}+0.32)   "
              f"end2end={t_cuda:.4f} ms   speedup={t_triton/t_cuda:.3f}x")
        row = dict(row, impl="cuda", ms=round(t_cuda, 4), cfused=round(t_cfused, 4),
                   speedup=round(t_triton / t_cuda, 4),
                   max_rel=mx, mean_rel=mn)

    with open(args.csv, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
