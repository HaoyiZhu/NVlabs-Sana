#!/usr/bin/env bash
set -euo pipefail

# Canonical single-GPU highest-quality Sana-WM streaming benchmark.
# Measures Stage-1 + refiner + causal VAE decode while excluding CPU frame
# transfer and MP4 encoding by default.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

STREAMING_ROOT="${STREAMING_ROOT:-pretrained_models/sana_wm_streaming}"
OUTPUT_MODE="${OUTPUT_MODE:-discard}"
OUT_DIR="${OUT_DIR:-benchmark_outputs/sana_wm_streaming_${OUTPUT_MODE}_$(date +%Y%m%d_%H%M%S)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-1}"
NUM_FRAMES="${NUM_FRAMES:-961}"
CAMERA_MODE="${CAMERA_MODE:-action}"
ACTION="${ACTION:-w-240,jw-120,w-240,lw-120,w-240}"
TRANSLATION_SPEED="${TRANSLATION_SPEED:-0.055}"
ROTATION_SPEED_DEG="${ROTATION_SPEED_DEG:-1.2}"
PROFILE_CUDA="${PROFILE_CUDA:-0}"
SAMPLE_FRAME_STRIDE="${SAMPLE_FRAME_STRIDE:-16}"
SAMPLE_FRAMES_NPZ="${SAMPLE_FRAMES_NPZ:-}"
SAVE_SAMPLE_FRAMES="${SAVE_SAMPLE_FRAMES:-0}"
export CUDA_VISIBLE_DEVICES
export DPM_TQDM="${DPM_TQDM:-True}"
export FUSED_GDN_PRECISION="${FUSED_GDN_PRECISION:-2}"

mkdir -p "${OUT_DIR}"
extra_args=()
case "${PROFILE_CUDA}" in
  1|true|TRUE|yes|YES) extra_args+=(--profile_cuda) ;;
esac
case "${SAVE_SAMPLE_FRAMES}" in
  1|true|TRUE|yes|YES)
    if [[ -z "${SAMPLE_FRAMES_NPZ}" ]]; then
      SAMPLE_FRAMES_NPZ="${OUT_DIR}/sampled_frames.npz"
    fi
    ;;
esac
if [[ -n "${SAMPLE_FRAMES_NPZ}" ]]; then
  extra_args+=(--sample_frames_npz "${SAMPLE_FRAMES_NPZ}" --sample_frame_stride "${SAMPLE_FRAME_STRIDE}")
fi

if [[ "${CAMERA_MODE}" == "camera" ]]; then
  camera_args=(
    --camera asset/sana_wm/demo_0_pose.npy
    --intrinsics asset/sana_wm/demo_0_intrinsics.npy
  )
elif [[ "${CAMERA_MODE}" == "action" ]]; then
  broadcast_intrinsics="${OUT_DIR}/demo_0_intrinsics_3x3.npy"
  python - "${broadcast_intrinsics}" <<'PY'
from pathlib import Path
import sys
import numpy as np

src = np.load("asset/sana_wm/demo_0_intrinsics.npy").astype(np.float32)
if src.ndim == 3:
    src = src[0]
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
np.save(sys.argv[1], src)
PY
  camera_args=(
    --action "${ACTION}"
    --translation_speed "${TRANSLATION_SPEED}"
    --rotation_speed_deg "${ROTATION_SPEED_DEG}"
    --intrinsics "${broadcast_intrinsics}"
  )
else
  echo "CAMERA_MODE must be 'action' or 'camera'; got '${CAMERA_MODE}'." >&2
  exit 2
fi

python inference_video_scripts/inference_sana_wm_streaming.py \
  --image asset/sana_wm/demo_0.png \
  --prompt asset/sana_wm/demo_0.txt \
  "${camera_args[@]}" \
  --num_frames "${NUM_FRAMES}" \
  --fps 16 \
  --cfg_scale 1.0 \
  --flow_shift 8.0 \
  --streaming_root "${STREAMING_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --name demo_0 \
  --output_mode "${OUTPUT_MODE}" \
  --benchmark_json "${OUT_DIR}/result.json" \
  --benchmark_repeats "${BENCHMARK_REPEATS}" \
  "${extra_args[@]}" \
  "$@"

echo "Wrote ${OUT_DIR}/result.json"
