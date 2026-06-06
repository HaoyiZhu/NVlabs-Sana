#!/usr/bin/env bash
# Launch the SANA-WM **realtime** interactive demo (Gradio + HLS preview).
#
# This is the realtime (feat/sana-wm-realtime) counterpart of
# scripts/launch_sana_wm_streaming_demo.sh. The app now AUTO-DETECTS Blackwell
# (sm_100 GB200/B200, sm_120 5090/GB10) and enables stage-1 NVFP4, refiner NVFP4
# and the fp8 KV cache only there; on Hopper/Ada it warns and falls back to bf16.
# No manual SANA_WM_REALTIME_BLACKWELL toggle is needed anymore.
#
# Usage:
#   ./scripts/launch_sana_wm_realtime_demo.sh                 # 0.0.0.0:7860 local
#   ./scripts/launch_sana_wm_realtime_demo.sh --share         # public *.gradio.live URL
#   ./scripts/launch_sana_wm_realtime_demo.sh --no_compile    # skip torch.compile (fast cold start)
set -euo pipefail

WT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WT"

# Editable sana-0.2.0 install shadows worktree imports to the main repo;
# pin imports to this (realtime) worktree.
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
export DISABLE_XFORMERS="${DISABLE_XFORMERS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[realtime-demo] WT=$WT (NVFP4 auto-enabled on Blackwell; bf16 fallback elsewhere)"
exec python app/app_sana_wm_realtime.py "$@"
