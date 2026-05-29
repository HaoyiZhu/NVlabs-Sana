#!/usr/bin/env bash
set -euo pipefail

# HSG Slurm wrapper for the canonical Sana-WM single-GPU benchmark.
# Run this inside an allocated GPU shell, typically:
#   srun -A nvr_elm_llm --qos=interactive -p batch --gres=gpu:4 \
#     --cpus-per-task=32 --mem=920G --time=04:00:00 --pty bash

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to run outside a Slurm allocation." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null; then
  echo "nvidia-smi failed; refusing to run benchmark." >&2
  exit 3
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV_PREFIX=""
for candidate in \
  /home/jinchengy/miniforge3/envs/sana-wm \
  /home/jinchengy/miniforge3/envs/sana \
  /home/jinchengy/miniconda3/envs/sana-wm \
  /home/jinchengy/miniconda3/envs/sana; do
  if [[ -x "${candidate}/bin/python" ]]; then
    CONDA_ENV_PREFIX="${candidate}"
    break
  fi
done
if [[ -n "${CONDA_ENV_PREFIX}" ]]; then
  export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export CONDA_PREFIX="${CONDA_ENV_PREFIX}"
  export CONDA_DEFAULT_ENV="$(basename "${CONDA_ENV_PREFIX}")"
else
  echo "Warning: could not find conda env sana-wm or sana; using current Python." >&2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export STREAMING_ROOT="${STREAMING_ROOT:-pretrained_models/sana_wm_streaming}"
export BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-2}"
export OUTPUT_MODE="${OUTPUT_MODE:-discard}"
export NUM_FRAMES="${NUM_FRAMES:-961}"
export FUSED_GDN_PRECISION="${FUSED_GDN_PRECISION:-2}"

OUT_DIR="${OUT_DIR:-benchmark_outputs/eac6_streaming_${NUM_FRAMES}_${OUTPUT_MODE}_$(date +%Y%m%d_%H%M%S)}"
export OUT_DIR
mkdir -p "${OUT_DIR}"

{
  printf '__CODEX_COMPUTE_GUARD__ host=%s user=%s slurm_job=%s cuda=%s\n' \
    "$(hostname)" "$(whoami)" "${SLURM_JOB_ID}" "${CUDA_VISIBLE_DEVICES}"
  printf '__CODEX_ENV__ python=%s conda=%s pwd=%s frames=%s repeats=%s output_mode=%s fused_gdn_precision=%s\n' \
    "$(command -v python)" "${CONDA_DEFAULT_ENV:-none}" "$(pwd)" "${NUM_FRAMES}" \
    "${BENCHMARK_REPEATS}" "${OUTPUT_MODE}" "${FUSED_GDN_PRECISION:-default}"
  printf '__CODEX_SAMPLE__ save=%s stride=%s path=%s\n' \
    "${SAVE_SAMPLE_FRAMES:-0}" "${SAMPLE_FRAME_STRIDE:-16}" "${SAMPLE_FRAMES_NPZ:-}"
  python - <<'PY'
import torch
print("__CODEX_TORCH__", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("__CODEX_GPU__", torch.cuda.get_device_name(0))
PY
  printf '__CODEX_BENCH_START__ out=%s\n' "${OUT_DIR}"
} | tee "${OUT_DIR}/run.log"

set +e
scripts/benchmark_sana_wm_streaming.sh "$@" >> "${OUT_DIR}/run.log" 2>&1
rc=$?
set -e
printf '__CODEX_BENCH_DONE__ rc=%s out=%s\n' "${rc}" "${OUT_DIR}" | tee -a "${OUT_DIR}/run.log"
exit "${rc}"
