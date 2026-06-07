#!/usr/bin/env python3
"""Isolated Phase-A runners for ncu profiling (phase_a_kv + cam_phase_a_kv)."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import make_inputs
from harness_cam import make_cam_inputs
import cuda_impl as ci

B, F, S, H, D = 1, 11, 920, 20, 128
dev = torch.device("cuda")
ext = ci.build()
BH = B * H

# ---- full-path phase_a_kv ----
inp = make_inputs(B, F, S, H, D, dev, torch.bfloat16, 0)
IPk = torch.empty(BH, F, 128, 128, device=dev, dtype=torch.bfloat16); A = torch.empty_like(IPk)
beta = inp["beta"].float(); kir = inp["k_inv_rms"].float(); knw = inp["k_norm_w"].float()
rc, rs = inp["rope_cos"], inp["rope_sin"]; qkv = inp["qkv"].contiguous()

# ---- cam-path cam_phase_a_kv ----
q, k, v, cbeta, cdecay = make_cam_inputs(B, F, S, H, D, dev, 0)
cIPk = torch.empty(BH, F, 128, 128, device=dev, dtype=torch.bfloat16); cA = torch.empty_like(cIPk)
cbeta_f = cbeta.contiguous().float(); kc = k.contiguous(); vc = v.contiguous(); qc = q.contiguous()
Mbf = torch.zeros(BH, F, 128, 128, device=dev, dtype=torch.bfloat16)
cout = torch.empty(B, H, D, F * S, device=dev, dtype=torch.float32)

for _ in range(30):
    ext.phase_a_kv(qkv, beta, kir, knw, rc, rs, IPk, A, F, S, 1.0)
    ext.cam_phase_a_kv(kc, vc, cbeta_f, cIPk, cA, F, S)
    ext.cam_phase_c(qc, Mbf, cout, F, S)
torch.cuda.synchronize()
print("done")
