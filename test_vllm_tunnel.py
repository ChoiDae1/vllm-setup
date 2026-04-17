from sshtunnel import SSHTunnelForwarder
import requests

# SSH 터널 설정
tunnel = SSHTunnelForwarder(
    ssh_address_or_host=("59.29.246.22", 20001),
    ssh_username="daewon",
    ssh_pkey="~/.ssh/id_rsa",           # 개인키 경로
    ssh_proxy_host="143.248.53.154",    # ProxyJump
    remote_bind_address=("node7", 8000),
    local_bind_address=("localhost", 8000),
)

tunnel.start()

try:
    base_url = f"http://localhost:{tunnel.local_bind_port}/v1"

    # 모델 확인
    r = requests.get(f"{base_url}/models")
    print(r.json())

    # 채팅
    payload = {
        "model": "gemma4-4b",
        "messages": [{"role": "user", "content": "안녕!"}],
        "max_tokens": 128,
    }
    r = requests.post(f"{base_url}/chat/completions", json=payload)
    print(r.json()["choices"][0]["message"]["content"])

finally:
    tunnel.stop()
