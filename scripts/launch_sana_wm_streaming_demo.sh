#!/usr/bin/env bash
# Launch the SANA-WM interactive streaming demo (FastAPI + WebSocket).
#
# Usage:
#   ./scripts/launch_sana_wm_streaming_demo.sh                    # 0.0.0.0:7860, local-only
#   ./scripts/launch_sana_wm_streaming_demo.sh --share            # also publish *.gradio.live URL
#   ./scripts/launch_sana_wm_streaming_demo.sh --port 17860
#   ./scripts/launch_sana_wm_streaming_demo.sh --streaming_root /path/to/weights
#
# First start pays the torch.compile cost (~3 min cold, ~30 s warm cache).
# After load, open http://localhost:<port>/ (or the printed *.gradio.live URL).

set -euo pipefail

cd "$(dirname "$0")/.."

export DISABLE_XFORMERS="${DISABLE_XFORMERS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python app/app_sana_wm_streaming.py "$@"
