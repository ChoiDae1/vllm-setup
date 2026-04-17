#!/bin/bash
#SBATCH --job-name=vllm-gemma4-4b-2
#SBATCH --partition=a6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/lustre/slurm/users/daewon/evospec/logs/gemma4_4b_%j.out
#SBATCH --error=/mnt/lustre/slurm/users/daewon/evospec/logs/gemma4_4b_%j.err

set -e

# ── 경로 설정 ─────────────────────────────────────────────────
LUSTRE_BASE="/mnt/lustre/slurm/users/daewon/evospec"
LOCAL_BASE="/tmp/daewon_${SLURM_JOB_ID}"
VENV="${LOCAL_BASE}/venv_gemma4"
HF_CACHE="${LUSTRE_BASE}/hf_cache"   # lustre 그대로
LOG_DIR="${LUSTRE_BASE}/logs"

VENV_PACK="${LUSTRE_BASE}/venv_gemma4.tar.gz"

mkdir -p "${LOCAL_BASE}" "${LOG_DIR}"

# ── HF 환경변수 (lustre) ──────────────────────────────────────
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN not set}"

# ── CUDA / NCCL ───────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1

# ── compile 캐시 NVMe로 ───────────────────────────────────────
export VLLM_COMPILE_CACHE_DIR="${LOCAL_BASE}/vllm_compile_cache"
mkdir -p "${VLLM_COMPILE_CACHE_DIR}"

# ── Slack 설정 ────────────────────────────────────────────────
SLACK_TOKEN="${SLACK_BOT_TOKEN:?SLACK_BOT_TOKEN not set}"
SLACK_DM="${SLACK_DM_CHANNEL:?SLACK_DM_CHANNEL not set}"

slack_notify() {
    local msg="$1"
    curl -s -X POST "https://slack.com/api/chat.postMessage" \
        -H "Authorization: Bearer ${SLACK_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"channel\": \"${SLACK_DM}\", \"text\": \"${msg}\"}" \
        > /dev/null 2>&1 || true
}

# ── venv 체크 및 압축 해제 ────────────────────────────────────
VENV_MTIME=$(stat -c %Y "$VENV_PACK" 2>/dev/null || echo 0)
VENV_LOCAL_MTIME=$(cat "$VENV/.packed_at" 2>/dev/null || echo 0)

if [ ! -f "$VENV/bin/python" ] || [ "$VENV_MTIME" -gt "$VENV_LOCAL_MTIME" ]; then
    echo "=== Unpacking venv to NVMe ==="
    rm -rf "$VENV"
    tar -xf "$VENV_PACK" -C "$LOCAL_BASE"
    echo "$VENV_MTIME" > "$VENV/.packed_at"
    echo "=== Patching shebangs ==="
    grep -rl "${LUSTRE_BASE}/venv_gemma4" "${VENV}/bin/" \
        | xargs sed -i "s|${LUSTRE_BASE}/venv_gemma4|${VENV}|g" 2>/dev/null || true
    echo "venv unpack done. $(date)"
else
    echo "=== venv up-to-date on NVMe ($(date -d @$VENV_LOCAL_MTIME)) ==="
fi

# ── venv 활성화 ───────────────────────────────────────────────
sed -i "s|VIRTUAL_ENV=.*|VIRTUAL_ENV=\"${VENV}\"|" "${VENV}/bin/activate"
source "${VENV}/bin/activate"

# ── 시작 로그 ─────────────────────────────────────────────────
NODE=$(hostname)
echo "=============================================="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : ${NODE}"
echo "GPUs       : ${CUDA_VISIBLE_DEVICES}"
echo "Model      : google/gemma-4-E4B-it"
echo "Port       : 8002"
echo "Start time : $(date)"
echo "=============================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

slack_notify ":rocket: *Gemma4 4B-2 vLLM 서버 시작*
- Job ID: \`${SLURM_JOB_ID}\`
- Node: \`${NODE}\`
- Endpoint: \`http://${NODE}:8002/v1\`
- 시작: $(date '+%Y-%m-%d %H:%M:%S')"

# ── 종료 트랩 (크래시 포함) ──────────────────────────────────
trap 'slack_notify ":stop_sign: *Gemma4 4B-2 서버 종료*  Job \`${SLURM_JOB_ID}\` | $(date +'"'"'%Y-%m-%d %H:%M:%S'"'"')"; rm -rf "${LOCAL_BASE}"' EXIT

# ── vLLM 서버 실행 ────────────────────────────────────────────
vllm serve google/gemma-4-E4B-it \
    --host 0.0.0.0 \
    --port 8002 \
    -tp 2 \
    --dtype auto \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --disable-custom-all-reduce \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --chat-template /home/daewon/evospec/chat_template_keep_thinking_v3.jinja \
    --served-model-name gemma4-4b-2

echo "Server stopped at $(date)"