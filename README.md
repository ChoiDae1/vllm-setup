# evospec — vLLM 서버 셋업

SLURM 클러스터에서 Gemma4 (4B / 31B) 및 Qwen3-Embedding-8B 를 vLLM OpenAI-호환 서버로 띄우는 스크립트 모음. NVMe 로컬 venv + Slack 알림 + speculative decoding 실험용 chat template 포함.

---

## 1. 사전 준비

### 클러스터 환경
- SLURM (priority=1, FIFO)
- GPU 파티션: `a6000` (8x A6000 48GB per node), `4090` (8x RTX4090 24GB)
- 공유 FS: `/mnt/lustre/slurm/users/<user>/`
- 각 노드에 `/tmp` 로컬 디스크 (NVMe, ~23GB free)

### 필수 계정·토큰
- HuggingFace token (Gemma4 gated model 접근용) → https://huggingface.co/settings/tokens
- Slack bot token (`chat:write`, `im:write` 스코프) — 알림 전송용
- Slack DM 채널 ID

---

## 2. 최초 1회 환경 구축

### 2-1. Python 환경 (vllm 0.19.0 + transformers 5.5.0)

**재현 가능한 정확한 버전 (권장)** — `requirements.lock.txt` 사용:

```bash
uv venv venv_gemma4 --python 3.12
source venv_gemma4/bin/activate
uv pip install -r requirements.lock.txt
```

> `requirements.lock.txt` 는 실제 동작하는 환경에서 `uv pip freeze` 로 동결한 174개 패키지 목록. vllm==0.19.0, transformers==5.5.0, torch==2.10.0+cu126 고정.

**처음부터 설치 (nightly drift 주의)**:

```bash
uv venv venv_gemma4 --python 3.12
source venv_gemma4/bin/activate

uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu126 \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match
uv pip install transformers==5.5.0
```

> 구버전 conda `agents` 환경은 Gemma4 미지원. 반드시 이 uv venv 사용.

### 2-2. 환경변수 설정

`.env.example` 를 복사해서 `.env` 작성:

```bash
cp .env.example .env
# .env 열어서 실제 값 기입
```

필요 변수:
| 키 | 용도 |
|---|---|
| `HF_TOKEN` | Gemma4 모델 다운로드 |
| `SLACK_BOT_TOKEN` | 서버 시작/종료 알림 |
| `SLACK_DM_CHANNEL` | 알림 받을 DM 채널 ID |

매 세션마다 로드:
```bash
set -a; source .env; set +a
```

### 2-3. venv 패키징 (NVMe 배포용)

job이 시작되면 `/tmp/daewon_${JOB_ID}/` 에 venv 를 풀어서 쓴다. 미리 tar.gz 로 묶어둬야 함:

```bash
tar -czf venv_gemma4.tar.gz venv_gemma4
```

→ 약 4.4GB. lustre에 두면 job 시작 시 NVMe 로 자동 압축 해제 (shebang 패치까지 스크립트가 처리).

### 2-4. HuggingFace 캐시

`hf_cache/` 에 모델 다운로드. 첫 job 실행 시 자동으로 받아짐 (lustre 직접 사용, NVMe 이동 안 함).

---

## 3. 서버 제출

```bash
# env 로드 (HF_TOKEN, SLACK_BOT_TOKEN, SLACK_DM_CHANNEL)
set -a; source .env; set +a

# 빈 노드 확인
sinfo -o "%P %N %T %G %C" --partition=a6000

# 제출
JOB_4B=$(sbatch --parsable slurm_gemma4_4b_nvme.sh)   # port 8000
JOB_31B=$(sbatch --parsable slurm_gemma4_31b_nvme.sh) # port 8001

# 같은 노드에 묶고 싶으면
JOB_4B=$(sbatch --parsable --nodelist=<node> slurm_gemma4_4b_nvme.sh)
JOB_31B=$(sbatch --parsable --nodelist=<node> slurm_gemma4_31b_nvme.sh)
```

`_2` 변형은 같은 노드에 두 세트를 동시에 올릴 때 사용 (ports 8002, 8003).

### 서버 구성 (공통)
- `-tp 2 --dtype auto --max-model-len 16384 --gpu-memory-utilization 0.9`
- `--enable-prefix-caching --disable-custom-all-reduce`
- `--chat-template chat_template_keep_thinking_v3.jinja`
- `--enable-auto-tool-choice --tool-call-parser gemma4`

| 스크립트 | 모델 | TP | Port |
|---|---|---|---|
| `slurm_gemma4_4b_nvme.sh` | google/gemma-4-E4B-it | 2 | 8000 |
| `slurm_gemma4_31b_nvme.sh` | google/gemma-4-31B-it | 2 | 8001 |
| `slurm_gemma4_4b_nvme_2.sh` | google/gemma-4-E4B-it | 2 | 8002 |
| `slurm_gemma4_31b_nvme_2.sh` | google/gemma-4-31B-it | 2 | 8003 |
| `slurm_qwen3_emb_8b.sh` | Qwen/Qwen3-Embedding-8B | 1 | 8002 |

---

## 4. 상태 확인

```bash
# 내 job 목록
squeue -u $USER -o "%.8i %.20j %.8T %.10L %N"

# 노드별 GPU 할당
for n in node3 node4 node5 node6 node7 node8 node9 node10; do
  echo "$n: $(scontrol show node $n | grep -oP 'AllocTRES=.*gres/gpu=\K[0-9]+' | head -1)/8"
done

# 로그 tail
tail -f logs/gemma4_4b_${JOB_4B}.out
```

서버 기동 확인:
```bash
NODE=$(squeue -j ${JOB_4B} -h -o %N)
curl http://${NODE}:8000/v1/models
```

---

## 5. 원격 접속 (SSH 터널)

```python
from sshtunnel import SSHTunnelForwarder

tunnel = SSHTunnelForwarder(
    ssh_address_or_host=("<bastion_ip>", <bastion_port>),
    ssh_username="<user>",
    ssh_pkey="~/.ssh/id_rsa",
    ssh_proxy_host="<proxy_ip>",
    remote_bind_address=("<nodename>", 8000),
    local_bind_address=("localhost", 8000),
)
```

노드 hostname 이 DNS 미등록이면 IP 직접 사용:
```bash
getent hosts <nodename>
```

---

## 6. Chat Template

| 파일 | 동작 |
|---|---|
| `chat_template_keep_thinking.jinja` (v1) | 모든 과거 assistant 턴의 `<think>` 유지 |
| `chat_template_keep_thinking_v2.jinja` | v1 + state reset 버그픽스 |
| **`chat_template_keep_thinking_v3.jinja`** | 원본처럼 과거 think strip + **마지막 메시지가 assistant 일 때만** think 유지 (prefill/continuation 용) |

기본값은 v3. 원본과 동일한 학습 분포 유지하면서 assistant prefill 시나리오만 지원.

---

## 7. 알려진 이슈

### vLLM 0.19.0 shm_broadcast hang
- TP=2 초기화 중 `custom_all_reduce` 가 worker를 60s+ block → EngineCore timeout → `shm_broadcast.py` 루프
- **해결: `--disable-custom-all-reduce`** (모든 스크립트에 적용됨)
- `VLLM_USE_V1=0` 는 0.19.0 에서 무시됨

### 강제 scancel 후 GPU 오염
- 다음 job 에서 NCCL "unspecified launch failure" 가능
- 재제출 시 해당 노드 `--exclude` 권장

### 첫 요청 지연
- vLLM 0.19.0 inductor 컴파일로 첫 요청 30s~1m 소요 (이후 정상)

---

## 8. 디렉토리 구조

```
evospec/
├── README.md                          ← 본 문서
├── CLAUDE.md                          ← 내부 운영 노트
├── .env.example                       ← 필요 환경변수 템플릿
├── requirements.lock.txt              ← 동결된 패키지 버전 (174개)
├── .gitignore
├── setup_vllm_env.sh                  ← (legacy) conda 기반 셋업
├── submit_vllm.sh                     ← 4B + 31B 동시 제출 헬퍼
├── slurm_gemma4_4b_nvme.sh            ← 4B (port 8000)
├── slurm_gemma4_31b_nvme.sh           ← 31B (port 8001)
├── slurm_gemma4_4b_nvme_2.sh          ← 4B secondary (port 8002)
├── slurm_gemma4_31b_nvme_2.sh         ← 31B secondary (port 8003)
├── slurm_qwen3_emb_8b.sh              ← Qwen3 embedding (port 8002)
├── chat_template_keep_thinking*.jinja ← chat template v1/v2/v3
├── test_vllm*.py                      ← 로컬/원격 테스트 클라이언트
├── venv_gemma4/                       ← (gitignored) Python venv
├── venv_gemma4.tar.gz                 ← (gitignored) NVMe 배포용
├── hf_cache/                          ← (gitignored) HF 모델 캐시
└── logs/                              ← (gitignored) SLURM 로그
```
