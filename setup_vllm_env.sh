#!/bin/bash
# ============================================================
# vLLM 환경 설정 스크립트 (Gemma4용)
# 최초 1회만 실행 필요
# ============================================================

set -e

CONDA_BASE="/mnt/lustre/slurm/users/daewon/miniforge3"
ENV_NAME="vllm"
ENV_PATH="${CONDA_BASE}/envs/${ENV_NAME}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 이미 환경이 있으면 스킵
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] conda env '${ENV_NAME}' already exists. Skipping creation."
else
    echo "[INFO] Creating conda env '${ENV_NAME}' with Python 3.12..."
    conda create -y -n "${ENV_NAME}" python=3.12
fi

conda activate "${ENV_NAME}"

echo "[INFO] Installing vLLM nightly (CUDA 12.9)..."
pip install -U pip
pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --index-strategy unsafe-best-match

echo "[INFO] vLLM version:"
python -c "import vllm; print(vllm.__version__)"

echo "[DONE] Setup complete. Env: ${ENV_PATH}"
echo ""
echo "HuggingFace 토큰 설정 (Gemma4는 gated model):"
echo "  export HF_TOKEN=<your_hf_token>"
echo "  huggingface-cli login --token \$HF_TOKEN"
