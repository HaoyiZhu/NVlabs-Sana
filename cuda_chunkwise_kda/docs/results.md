# Results — Chunkwise GDN Triton → CUDA (RTX 5090 / sm_120)

All numbers: RTX 5090 (sm_120), torch 2.11+cu129, bf16 (dot_prec=0), B=1, F=11,
S=920, H=20, D=128. Median of CUDA-event timings. Correctness oracle = Triton
output on identical inputs.

## Full BiGDN path (`fused_bigdn_bidi_chunkwise`)
Triton end2end **0.837 ms** = phaseA 0.337 + phaseB 0.061 + phaseC 0.110 + **divide 0.32**.
Key insight: the Python-side `num/(den+eps)` divide (fp32, 26M elts, permute) is
38% of B=1 and 69% of B=4 — not a compute cost, a glue cost.

| config | end2end | speedup | notes |
|---|---|---|---|
| Triton (all) | 0.837 ms | 1.00× | baseline |
| Triton A/B + **CUDA C+divide fused** | **0.669 ms** | **1.25×** | divide fused into Phase C (regs), max_rel 1.39e-2 |

CUDA Phase A (kv+z): bit-exact vs Triton (max_abs=0) but 0.97 ms vs Triton's
well-tuned 0.337 ms — Triton's prep+GEMM is near-optimal here, so Phase A stays
Triton in the perf config (`--cmode c`). Full-CUDA A+C (`--cmode ac`) is correct
but slower (Phase A dominates). CUDA Phase A exists for full-rewrite completeness.

## Cam path (`cam_scan_bidi_chunkwise`) — the LIVE model entry
Identity norm/RoPE, skip_relu, skip_z (num-only), output transposed to [B,H,D,N].
Triton end2end **4.067 ms**, but the math (phaseA 0.376 + B 0.041 + C 0.135) is
only ~0.55 ms — the rest is glue: packing q/k/v into [B,N,3,H,D] (3× 104MB
permute-copies) + output permute/contiguous (0.36 ms). A CUDA kernel reading
q/k/v [B,H,D,N] directly and writing the transposed fp32 output directly
eliminates all of it.

| config | end2end | speedup | notes |
|---|---|---|---|
| Triton cam | 4.068 ms | 1.00× | baseline (incl. packing + transpose glue) |
| **CUDA cam** (A+C direct, Triton B) | **0.872 ms** | **4.67×** | reads q/k/v [B,H,D,N] direct, writes transposed fp32 direct; max_rel 3.9e-3 |

Cam B-sweep (all PASS, max_rel 3.9e-3): B=1 4.07→0.87ms **4.67×**, B=4 13.54→7.91ms
**1.71×**, B=8 27.02→13.49ms **2.00×**. (B=1 = the realtime single-stream case.
At higher B the compute-bound Phase A GEMM dominates and narrows the gap.)

## Summary (validated on RTX 5090, sm_120)
- **Full BiGDN path: 1.25×** (0.835→0.669 ms), max_rel 1.39e-2.
- **Cam path (live model): 4.67×** (4.07→0.87 ms, B=1/H=20), max_rel 3.9e-3.
- Both numerically validated against the Triton output; CUDA Phase A is bit-exact.

## Reproduce
```
conda activate svideo   # on x5
# full path:  Triton baseline + CUDA C-fused (best config)
python cuda_chunkwise_kda/harness.py --impl cuda --cmode c
# cam path (production):
python cuda_chunkwise_kda/harness_cam.py
# Phase A bit-exactness:
python cuda_chunkwise_kda/test_phase_a.py
```
Drop-in for the model: `cuda_impl.cam_scan_bidi_chunkwise_cuda` (CUDA for D=128
fp32 contiguous, Triton fallback otherwise).

## Not pursued / future work
- CUDA Phase A is correct (bit-exact) but ~3× slower than Triton's tuned GEMM, so
  the perf config keeps Triton's Phase A. A split-K / cp.async pipelined Phase A
  could close it.
- A CUDA cam Phase B (to drop the fp32 M_hist→bf16 copy) needs **fp32** M-state
  (96 KB smem opt-in); the bf16-state attempt drifted (max_rel 0.14) and was slow.

## Build note
svideo conda nvcc ships without usable headers; pip CUDA headers are fragmented.
Build against the full system toolkit `/usr/local/cuda-12.9` (see cuda_impl.build).

## Key CUDA design choices (sm_120 / Blackwell consumer)
- bf16 `wmma` m16n16k16 fragments (NOT fp16) to match Triton numerics.
- M kept OUT of smem: bf16, loaded into wmma b-frags directly from global (L2-
  resident ~14MB) → occupancy set by tiny Q working set, not the 32KB M.
- Grid (BH*F, S_TILES) for Phase C: many CTAs to fill the 170 SMs (vs BH=20).
- Phase A: 16 warps so each owns one output's row-tile (8 frags) — avoids the
  register spill that 16 frags/warp caused (3.6ms → 0.97ms).
- Fuse the memory-bound glue (divide / transpose) into the compute kernel — the
  single biggest win, and exactly what Triton's per-op Python wrapper can't do.
