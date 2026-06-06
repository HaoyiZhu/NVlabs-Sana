#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SANA environment installer. Single source of truth for deps is
# pyproject.toml; this script only handles what can't live there: the conda env
# + Python 3.11, the arch-matched CUDA toolkit and torch wheels, and the few
# packages that need special install flags (mmcv / flash-attn / Pi3 /
# Transformer Engine).
#
# The CUDA flavour is selected by host architecture:
#   x86_64  (H100 / Ada)              -> torch cu128 + CUDA 12.8
#   aarch64 (GB200 / Grace-Blackwell) -> torch cu130 + CUDA 13.0
#
# Usage:
#   bash ./environment_setup.sh sana   # create a fresh conda env
#   bash ./environment_setup.sh        # install into the active env
#
# Idempotent: re-running on an existing env will reconcile versions.
# -----------------------------------------------------------------------------
set -e

# Check if we should skip environment setup entirely (used by CI).
if [ "${SKIP_ENV_SETUP}" = "true" ]; then
    echo "SKIP_ENV_SETUP is set to true. Skipping all environment setup steps."
    echo "Using default conda environment. Make sure it has all required packages installed."
    exit 0
fi

# Architecture-matched CUDA: Grace-Blackwell (aarch64) ships only cu130 wheels.
case "$(uname -m)" in
    aarch64) TORCH_CUDA=cu130; CUDA_TOOLKIT=13.0 ;;
    *)       TORCH_CUDA=cu128; CUDA_TOOLKIT=12.8 ;;
esac
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    eval "$(conda shell.bash hook)"

    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        echo "[sana] conda env '$CONDA_ENV' already exists; reusing it."
    else
        # Python 3.11+ required: triton 3.5's @triton.jit uses inspect.getsource
        # and regex-matches ``^def\s+\w+\s*\(``; on 3.10 the source returned for
        # fla's decorated kernels starts after the decorator line and the regex
        # returns None.
        conda create -n "$CONDA_ENV" python=3.11 -y
    fi
    conda activate "$CONDA_ENV"

    # nvcc must match the torch wheels' CUDA major for from-source builds
    # (flash-attn, Transformer Engine). torch ships its own runtime libs.
    conda install -c nvidia "cuda-toolkit=${CUDA_TOOLKIT}" -y
else
    echo "[sana] Skipping conda env creation. Make sure the target env is activated."
fi

# setuptools<80: mmcv 1.7.2's setup.py imports ``pkg_resources``, which
# setuptools 80+ no longer ships as an importable module.
pip install -U pip wheel
pip install "setuptools<80"

# Pre-install the torch stack from the arch-matched index. Versions match
# pyproject pins, so the subsequent ``pip install -e .`` treats them as
# satisfied. xformers ships x86_64-only wheels and SANA-WM runs with
# DISABLE_XFORMERS=1, so it is installed only there (and is gated by the same
# platform marker in pyproject.toml).
pip install --upgrade --index-url "$TORCH_INDEX" \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
[ "$TORCH_CUDA" = "cu128" ] && \
    pip install --upgrade --index-url "$TORCH_INDEX" xformers==0.0.33.post2

# mmcv must build without PEP 517 isolation so its setup.py sees the env's
# pre-installed torch + setuptools<80.
pip install --no-build-isolation mmcv==1.7.2

# Editable install resolves everything else from pyproject.toml.
pip install -e .

# Pi3X (camera intrinsics from a single image, used by SANA-WM): --no-deps so
# it doesn't downgrade torch/numpy.
pip install git+https://github.com/yyfz/Pi3.git --no-deps

# flash-attn (refiner attention), built against the env's torch + nvcc.
MAX_JOBS=${MAX_JOBS:-8} NVCC_THREADS=${NVCC_THREADS:-2} \
    pip install --no-build-isolation "flash-attn>=2.7.0"

# Transformer Engine powers the NVFP4 realtime path (Blackwell only). Prebuilt
# wheels exist for x86_64; on aarch64 it builds from source against the env torch.
MAX_JOBS=${MAX_JOBS:-8} \
    pip install --no-build-isolation "transformer_engine[pytorch]"

echo
echo "[sana] Done (${TORCH_CUDA}). Activate with:  conda activate ${CONDA_ENV:-<your-env>}"
