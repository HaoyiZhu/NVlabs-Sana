#!/usr/bin/env python3
"""Blackwell FP8 viability probe for the cam path.

sm_120's 5th-gen tensor cores do FP8 (e4m3/e5m2) at 2x bf16 and with half the
operand bytes (helps the smem-latency-bound cam_phase_a). But FP8 e4m3 has ~3
mantissa bits. This probe quantizes the cam GEMM operands to e4m3 (round-trip)
and measures the end-output error vs the bf16 path — tells us if an FP8 cam
kernel could stay within tolerance BEFORE writing one.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cuda.matmul.allow_tf32 = False
from diffusion.model.ops.fused_gdn_chunkwise import cam_scan_bidi_chunkwise
from harness_cam import make_cam_inputs


def q8(t, dt=torch.float8_e4m3fn):
    return t.to(dt).to(torch.float32)


def relerr(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs(); den = b.abs().clamp_min(1e-3)
    return (d / den).max().item(), (d / den).mean().item()


def main():
    B, F, S, H, D = 1, 11, 920, 20, 128
    dev = torch.device("cuda")
    q, k, v, beta, decay = make_cam_inputs(B, F, S, H, D, dev, 0)
    ref = cam_scan_bidi_chunkwise(q, k, v, beta, decay)  # bf16-dot baseline

    # FP8 on the K-stream operands (P_kv=K^T b K, A=K^T b V): quantize k, v.
    out_kv8 = cam_scan_bidi_chunkwise(q, q8(k), q8(v), beta, decay)
    # FP8 also on q (num = Q @ M): quantize q too.
    out_all8 = cam_scan_bidi_chunkwise(q8(q), q8(k), q8(v), beta, decay)
    # For reference: what bf16-rounding the inputs costs (the kernels already
    # cast operands to bf16 internally, so this ~ the current path's noise).
    out_bf16in = cam_scan_bidi_chunkwise(q.bfloat16().float(), k.bfloat16().float(),
                                         v.bfloat16().float(), beta, decay)

    print("# tolerance bar: max_rel <= 3e-2")
    for name, o in [("bf16-inputs", out_bf16in), ("fp8 k,v", out_kv8), ("fp8 q,k,v", out_all8)]:
        mx, mn = relerr(o, ref)
        print(f"#   {name:14s}: max_rel={mx:.3e}  mean_rel={mn:.3e}  "
              f"{'PASS' if mx <= 3e-2 else 'FAIL'}")


if __name__ == "__main__":
    main()
