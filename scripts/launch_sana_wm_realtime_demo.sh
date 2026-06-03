#!/usr/bin/env bash
# Launch the SANA-WM **realtime** interactive demo (Gradio + HLS preview).
#
# This is the realtime (feat/sana-wm-realtime) counterpart of
# scripts/launch_sana_wm_streaming_demo.sh. The app hard-codes Blackwell
# (GB200/5090) defaults via os.environ.setdefault (stage-1 NVFP4, refiner
# NVFP4, fp8 KV cache). Those abort on non-Blackwell GPUs, so this launcher
# force-disables them unless SANA_WM_REALTIME_BLACKWELL=1 is exported.
#
# Usage:
#   ./scripts/launch_sana_wm_realtime_demo.sh                 # 0.0.0.0:7860 local
#   ./scripts/launch_sana_wm_realtime_demo.sh --share         # public *.gradio.live URL
#   ./scripts/launch_sana_wm_realtime_demo.sh --no_compile    # skip torch.compile (fast cold start)
#   SANA_WM_REALTIME_BLACKWELL=1 ./scripts/launch_sana_wm_realtime_demo.sh   # keep NVFP4/fp8 (GB200/5090)
set -euo pipefail

WT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WT"

# Editable sana-0.2.0 install shadows worktree imports to the main repo;
# pin imports to this (realtime) worktree.
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
export DISABLE_XFORMERS="${DISABLE_XFORMERS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${SANA_WM_REALTIME_BLACKWELL:-0}" != "1" ]]; then
  # H100 / non-Blackwell safe: disable NVFP4 (4-bit, Blackwell-only) and fp8 KV.
  export SANA_WM_STAGE1_NVFP4=0
  export SANA_WM_STAGE1_NVFP4_MODE=""
  export SANA_WM_REFINER_NVFP4=0
  export SANA_WM_REFINER_KV_CACHE_DTYPE=bf16
  export SANA_WM_TE_NVFP4_CPU_STAGING=0
  echo "[realtime-demo] non-Blackwell GPU: NVFP4/fp8 disabled (export SANA_WM_REALTIME_BLACKWELL=1 to keep them)."
fi

echo "[realtime-demo] WT=$WT"
exec python app/app_sana_wm_realtime.py "$@"
