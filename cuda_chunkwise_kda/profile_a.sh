#!/usr/bin/env bash
# Run with: sudo bash cuda_chunkwise_kda/profile_a.sh
# Profiles phase_a_kv + cam_phase_a_kv with ncu (needs root for GPU perf counters).
# Reuses xieenze's torch-extension build cache (HOME) so it does NOT recompile.
set -e
source /home/xieenze/miniconda3/etc/profile.d/conda.sh
conda activate svideo
export HOME=/home/xieenze
REPO=/home/xieenze/jinchengy/code/NVlabs-Sana-wm5090
NCU=/usr/local/cuda-12.9/bin/ncu
"$NCU" --launch-skip 40 --launch-count 2 \
  -k "regex:phase_a_kv|cam_phase_a_kv" \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section WarpStateStats --section SchedulerStats \
  python "$REPO/cuda_chunkwise_kda/prof_a.py" 2>&1 | tee /tmp/ncu_a.txt
echo "=== wrote /tmp/ncu_a.txt ==="
