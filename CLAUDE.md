# CLAUDE.md — evospec 프로젝트 설정

## 클러스터 환경
- 스케줄러: SLURM (전체 priority=1, FIFO 방식)
- GPU 파티션:
  - `a6000`: 8개 노드(node3~10), 노드당 8x A6000 48GB → 총 64개
  - `4090`: 1개 노드(node2), 8x RTX 4090 24GB (node1은 debug 파티션)
  - `all`: 전체 노드 통합 파티션
- 공유 파일시스템: `/mnt/lustre/slurm/users/daewon/`
- Conda base: `/mnt/lustre/slurm/users/daewon/miniforge3`
- CUDA: 12.6 (cu126), torch 2.7.0+cu126
- **vLLM 환경: `venv_gemma4`** (uv venv, vllm 0.19.0 + transformers 5.5.0)
  - 경로: `/mnt/lustre/slurm/users/daewon/evospec/venv_gemma4`
  - 활성화: `source /mnt/lustre/slurm/users/daewon/evospec/venv_gemma4/bin/activate`
  - 설치: `uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu126 --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match && uv pip install transformers==5.5.0`
  - ~~`agents` conda 환경 (vllm 0.9.1)은 Gemma4 미지원~~
- `/tmp`는 NVMe 전용 마운트가 아닌 루트 디스크 (98GB, 여유 ~23GB on 로그인 노드)

## vLLM 서버 설정
- 모델: `google/gemma-4-E4B-it` (4B, MoE sparse), `google/gemma-4-31B-it` (31B, dense)
- 공통 옵션: `-tp 2 --dtype auto --max-model-len 8192 --gpu-memory-utilization 0.8 --enable-prefix-caching --disable-custom-all-reduce`
- 포트: 4B → 8000, 31B → 8001
- **두 서버 같은 노드에 배치 권장** (speculative decoding 실험 통제 목적)
- HF 캐시: `/mnt/lustre/slurm/users/daewon/evospec/hf_cache` (lustre 직접 사용)

## vLLM 0.19.0 알려진 이슈
- **shm_broadcast hang (4B/31B 공통)**: vLLM 0.19.0 V1 엔진(EngineCore)이 TP=2에서 초기화 후 `shm_broadcast.py` "No available shared memory broadcast block" 루프에 빠짐
  - **원인: `custom_all_reduce`** — CUDA graph와 무관하게 재현됨
  - custom_all_reduce 초기화 과정에서 worker가 60초 이상 block → EngineCore 타임아웃
  - **해결책: `--disable-custom-all-reduce`** (두 스크립트 모두 적용)
  - `VLLM_USE_V1=0` 은 0.19.0에서 무시됨 (V1 엔진 강제 활성화)
- **node3 / node5**: 과거 hang·NCCL crash 이력 있었으나 2026-04-17 시점 정상 동작 확인 — 재발 시에만 exclude
- **강제 scancel 후 GPU 오염**: 다음 job에서 NCCL "unspecified launch failure" 발생 가능 → 재제출 시 해당 노드 exclude
- **첫 요청 지연**: vLLM 0.19.0 inductor 컴파일 모드로 인해 첫 번째 요청이 30초~1분 소요될 수 있음 (이후 요청은 정상)

## NVMe venv 전략
- `venv_gemma4.tar.gz` (4.4GB)를 job 시작 시 `/tmp/daewon_${SLURM_JOB_ID}/`에 압축 해제
- **shebang 패치 필수**: bin/ 내 스크립트 shebang이 lustre 경로 하드코딩 → 이동 후 반드시 패치
  ```bash
  grep -rl "${LUSTRE_BASE}/venv_gemma4" "${VENV}/bin/" \
      | xargs sed -i "s|${LUSTRE_BASE}/venv_gemma4|${VENV}|g" 2>/dev/null || true
  ```
- job ID를 경로에 포함 (`/tmp/daewon_${SLURM_JOB_ID}`)하여 동시 job 간 충돌 방지
- job 종료 시 `trap EXIT`으로 `/tmp/daewon_${SLURM_JOB_ID}` 자동 정리 (epilog 미지원)
- hf_cache는 NVMe 이동 없이 lustre 직접 사용

## Job 제출 방법
```bash
# 같은 노드에 배치하려면 먼저 빈 노드 확인 후 --nodelist 지정
squeue -o "%.8i %.10u %b %.8T %.10L %N" --partition=a6000
scontrol show node <nodename> | grep AllocTRES

# 제출 (현재 모든 노드 사용 가능 — 문제 발생 시에만 --exclude)
JOB_4B=$(sbatch --parsable slurm_gemma4_4b_nvme.sh)
JOB_31B=$(sbatch --parsable slurm_gemma4_31b_nvme.sh)

# 같은 노드 지정 시
JOB_4B=$(sbatch --parsable --nodelist=<node> slurm_gemma4_4b_nvme.sh)
JOB_31B=$(sbatch --parsable --nodelist=<node> slurm_gemma4_31b_nvme.sh)
```

## 클러스터 상태 확인
```bash
# 파티션별 노드/GPU 상태
sinfo -o "%P %N %T %G %C" --partition=a6000

# GPU 실제 할당 현황
scontrol show node <nodename> | grep -E "Gres|CfgTRES|AllocTRES"

# 전체 job queue
squeue -o "%.8i %.10u %b %.8T %.10L %N" --partition=a6000
```

## SSH 터널 (원격 접속)
```python
from sshtunnel import SSHTunnelForwarder

tunnel = SSHTunnelForwarder(
    ssh_address_or_host=("59.29.246.22", 20001),
    ssh_username="daewon",
    ssh_pkey="~/.ssh/id_rsa",
    ssh_proxy_host="143.248.53.154",
    remote_bind_address=("<nodename>", 8000),  # 4B: 8000, 31B: 8001
    local_bind_address=("localhost", 8000),
)
```
- node hostname이 DNS 미등록 시 IP 사용: `getent hosts <nodename>`으로 확인
- node8 IP: `192.168.0.29`

## Slack 알림
- 봇 토큰: 환경변수 `SLACK_BOT_TOKEN` 에서 로드 (`.env` 참고)
- DM 채널 ID: 환경변수 `SLACK_DM_CHANNEL`
- 알림 시점: job 시작 / 종료 (크래시 포함, `trap EXIT` 사용)
- Slack 알림은 채널이 아닌 DM으로 전송 (봇 스코프: `chat:write`, `im:write`)

## 스크립트 위치
```
evospec/
├── CLAUDE.md
├── setup_vllm_env.sh
├── slurm_gemma4_4b_nvme.sh  # 4B Job 제출용 (port 8000)
├── slurm_gemma4_31b_nvme.sh # 31B Job 제출용 (port 8001)
├── venv_gemma4/             # uv venv — vllm 0.19.0 + transformers 5.5.0
├── venv_gemma4.tar.gz       # NVMe 압축 해제용 (4.4GB)
├── hf_cache/                # HuggingFace 모델 캐시
└── logs/                    # SLURM 출력 로그
```

## 주의사항
- SLURM job은 같은 노드 내 GPU 0,1 사용 (`CUDA_VISIBLE_DEVICES=0,1`)
- 31B 모델은 bf16 기준 ~62GB → TP=2 필수
- `set -e` + `trap EXIT`으로 에러/크래시 시 Slack 종료 알림 + NVMe 정리
- 4090 파티션(node2)은 GPU S:0-1 소켓 분산 구조라 2개 연속 할당이 막힐 수 있음
