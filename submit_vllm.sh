#!/bin/bash
# ============================================================
# Gemma4 4B + 31B vLLM 서버 SLURM 제출 스크립트
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── HF 토큰 확인 ────────────────────────────────────────────
if [[ -z "${HF_TOKEN}" ]]; then
    echo "[WARN] HF_TOKEN이 설정되지 않았습니다."
    echo "       Gemma4는 gated model이므로 토큰이 필요합니다."
    echo "       export HF_TOKEN=hf_xxxx 후 재실행하거나 스크립트에 직접 입력하세요."
    echo ""
fi

# ── vllm conda 환경 확인 ────────────────────────────────────
CONDA_BASE="/mnt/lustre/slurm/users/daewon/miniforge3"
if ! conda env list 2>/dev/null | grep -q "^vllm "; then
    echo "[ERROR] conda env 'vllm'이 없습니다. 먼저 setup_vllm_env.sh를 실행하세요:"
    echo "        bash ${SCRIPT_DIR}/setup_vllm_env.sh"
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/logs"

# ── 잡 제출 ─────────────────────────────────────────────────
echo "[INFO] Gemma4 4B 서버 제출 중..."
JOB_4B=$(sbatch --parsable "${SCRIPT_DIR}/slurm_gemma4_4b.sh")
echo "       Job ID: ${JOB_4B} (port 8000)"

echo "[INFO] Gemma4 31B 서버 제출 중..."
JOB_31B=$(sbatch --parsable "${SCRIPT_DIR}/slurm_gemma4_31b.sh")
echo "       Job ID: ${JOB_31B} (port 8001)"

echo ""
echo "=============================================="
echo " 제출 완료"
echo "=============================================="
echo " 4B  job : ${JOB_4B}  →  port 8000"
echo " 31B job : ${JOB_31B}  →  port 8001"
echo ""
echo " 상태 확인:"
echo "   squeue -j ${JOB_4B},${JOB_31B}"
echo ""
echo " 로그 확인:"
echo "   tail -f ${SCRIPT_DIR}/logs/gemma4_4b_${JOB_4B}.out"
echo "   tail -f ${SCRIPT_DIR}/logs/gemma4_31b_${JOB_31B}.out"
echo ""
echo " 서버 테스트 (노드 IP 확인 후):"
echo "   NODE_4B=\$(squeue -j ${JOB_4B} -h -o %N)"
echo "   curl http://\${NODE_4B}:8000/v1/models"
echo ""
echo "   NODE_31B=\$(squeue -j ${JOB_31B} -h -o %N)"
echo "   curl http://\${NODE_31B}:8001/v1/models"
echo "=============================================="
