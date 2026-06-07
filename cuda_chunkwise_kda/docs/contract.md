# Contract — Chunkwise GDN: Triton → CUDA (RTX 5090 / sm_120)

## Objective
Reimplement the chunkwise Gated-Delta-Net forward (`fused_bigdn_bidi_chunkwise`
in `diffusion/model/ops/fused_gdn_chunkwise.py`) in CUDA so that, on an
**RTX 5090 (sm_120, Blackwell consumer)**, the end-to-end forward is **faster
than the existing Triton implementation** at the production shape, with matching
numerics.

## Target shape (production)
- B = 1 (also validate 4, 8), F = 11 frames, S = 920 tokens/frame,
  H = 20 heads, D = 128 head-dim. N = F*S = 10120, BH = B*H = 20.
- dtype bf16 (dot_precision = 0). Inter-phase bridge fp32 where Triton uses fp32.

## Pipeline (must reproduce exactly)
1. Phase A (grid BH·F): per (bh,f), reduce over S → `I-P_kv`,`A` [D,D],
   `I-P_z` [D,D], `B` [D]. RMSNorm(Q/K) · norm_w, ReLU, RoPE (K_rot), β, V.
2. Phase B (grid BH, serial over F): `M = g·(I-P_kv)@M + A`,
   `z = g·(I-P_z)·z + B`; forward + reverse, combined → `M_hist`,`z_hist`.
3. Phase C (grid BH·F): `num = Q_rot @ M_hist`, `den = Σ Q·z_hist`.
4. Divide: `out = num / (den + eps)`.

## Validation command
`python cuda_chunkwise_kda/harness.py --impl cuda --check`
- Correctness oracle = the Triton path on identical inputs.
- Pass if `max_rel_err <= 2e-2` and `mean_rel_err <= 2e-3` (bf16 tolerance),
  no NaN/Inf.

## Promotion criteria
- Correctness passes AND median end-to-end CUDA fwd time < median Triton fwd
  time on the 5090 at B=1 (and not slower at B=4,8).
- Record every candidate in `benchmark.csv` + `candidates.jsonl`.

## Blackwell (sm_120) optimization levers to try
- `mma.sync.m16n8k16` bf16 warp-MMA (vs legacy `wmma` m16n16k16).
- `cp.async` (cp.async.cg) double/triple-buffered SMEM staging; TMA bulk copy.
- Fuse phases to cut kernel launches (B1 is launch/occupancy bound).
- Persistent producer→consumer megakernel for B→C overlap (fill idle SMs;
  only BH=20 producer CTAs otherwise leave ~150 SMs idle on 5090).
- 256 KB-ish dynamic SMEM opt-in (sm_120 max ~100 KB/SM — confirm), regs tuning.
