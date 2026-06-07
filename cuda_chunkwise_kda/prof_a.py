#!/usr/bin/env python3
"""Isolated phase_a_kv runner for ncu profiling."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import make_inputs
import cuda_impl as ci

B, F, S, H, D = 1, 11, 920, 20, 128
dev = torch.device("cuda")
inp = make_inputs(B, F, S, H, D, dev, torch.bfloat16, 0)
ext = ci.build()
BH = B * H
IPk = torch.empty(BH, F, 128, 128, device=dev, dtype=torch.bfloat16); A = torch.empty_like(IPk)
beta = inp["beta"].float(); kir = inp["k_inv_rms"].float(); knw = inp["k_norm_w"].float()
rc, rs = inp["rope_cos"], inp["rope_sin"]; qkv = inp["qkv"].contiguous()
for _ in range(30):
    ext.phase_a_kv(qkv, beta, kir, knw, rc, rs, IPk, A, F, S, 1.0)
torch.cuda.synchronize()
print("done")
