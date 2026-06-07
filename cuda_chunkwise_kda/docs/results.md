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

CUDA Phase A (kv+z): bit-exact vs Triton (max_abs=0). Optimized kv
**0.524 → 0.356 ms (~32%)** via single-pass prep (recompute the RoPE-pair `d^1`
from global instead of a 2nd pass+sync), `cp.async` double-buffered K/V prefetch,
hoisted per-row constants (invrms/beta) out of the d-loop, and float4-vectorized
cos/sin/normw loads. Still ~2× Triton's autotuned GEMM for this small-output
(128×128) / large-K (920) shape — the residual gap is Triton's codegen maturity
(reg alloc, scheduling); precise diagnosis is blocked because ncu has no
perf-counter permission on this shared box. Since the GEMM has no *structural*
CUDA advantage, the full-path perf config keeps Triton's Phase A (`--cmode c`).
(A 32-warp/4-frag variant raised occupancy but hurt the cam path's L1-cached
strided loads, so 16 warps/8 frags is used.)

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
| **CUDA cam** (A+C direct, Triton B) | **0.490 ms** | **8.30×** | reads q/k/v [B,H,D,N] direct, writes transposed fp32 direct, d-major coalesced staging + col-major coalesced output (ncu-guided); max_rel 3.9e-3 |

Cam B-sweep (all PASS, max_rel 3.9e-3): B=1 4.07→0.87ms **4.67×**, B=4 13.54→7.91ms
**1.71×**, B=8 27.02→13.49ms **2.00×**. (B=1 = the realtime single-stream case.)

**ncu-guided coalescing (root ncu; the box's perf-counters need admin):** ncu showed
cam_phase_a was L1TEX-latency bound — k/v are `[B,H,D,N]` (d-major), so s-major
staging read them strided (uncoalesced, ~64% L1). Staging **d-major** (read along N,
coalesced; transpose moved to the wmma b-operand) cut cam_phase_a 0.528→0.270ms.
ncu on cam_phase_c then showed it memory-bound (79% mem, MIO stalls) from the
scattered d-major **output** write — storing the wmma tile col-major makes the write
coalesced. **cam path B=1: 0.871 → 0.620 → 0.490 ms = 4.67× → 6.57× → 8.30×**
(cam_phase_a 0.528→0.270, cam_phase_c 0.293→0.163; all max_rel 3.9e-3 PASS).

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

## Exploration: CUDA-only & Blackwell-specific optimizations
- **CUDA-only wins already captured** (things Triton's per-op Python wrapper
  cannot do): fuse the `num/(den+eps)` divide into Phase C (full path); read
  q/k/v `[B,H,D,N]` directly + write the transposed fp32 output directly (cam),
  removing ~3.5ms of packing/transpose glue; d-major coalesced staging +
  col-major coalesced output. These ARE the CUDA-unique speedups.
- **Blackwell FP8/FP4 (5th-gen TC headline) — NOT viable here.** `fp8_probe.py`:
  e4m3-quantizing the cam GEMM operands gives **7.9–9.6% mean output error** vs
  **1.0%** for bf16 (bar = 3%). The delta-rule `(I−βkkᵀ)` is a near-identity
  contraction, so fp8's ~3-bit mantissa compounds over the F=11 frame recurrence.
  → bf16 is required; the right Blackwell primitive is the bf16 5th-gen TC (used).
- **sm_120 (consumer Blackwell) device facts:** 170 SMs, 48KB default / 100KB
  opt-in smem, 64K regs/SM, 1536 thr/SM, **no thread-block clusters / tensor
  memory** (those are sm_100 datacenter only). So distributed-smem megakernels
  aren't available; occupancy is reg+smem bound (8 wmma acc-frags = 64 regs is
  the floor → 33% occ is the cam_phase_a sweet spot; raising it spills or slows).
- **Remaining CUDA-only frontier:** a persistent producer→consumer megakernel
  fusing Phase B (serial scan) + Phase C (the `dev/v2v/bench_phase_bc_persistent_*`
  POC direction) — removes the Phase-B launch + fp32 `M_hist` HBM roundtrip and
  overlaps B's under-utilized serial scan with C's throughput. Quantified upside
  on cam is small (~0.06 ms of 0.49 ms ≈ 1.1×) at high complexity/risk; offered.

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
