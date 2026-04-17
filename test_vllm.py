import requests
import json

BASE_URL = "http://node7:8000/v1"

# 모델 목록 확인
print("=== Models ===")
r = requests.get(f"{BASE_URL}/models")
print(json.dumps(r.json(), indent=2))

# 채팅 테스트
print("\n=== Chat Completion ===")
payload = {
    "model": "gemma4-4b",
    "messages": [
        {"role": "user", "content": "안녕! 간단하게 자기소개 해줘."}
    ],
    "max_tokens": 256,
    "temperature": 1.0,
    "top_p": 0.95,
}

r = requests.post(f"{BASE_URL}/chat/completions", json=payload)
resp = r.json()
print(resp["choices"][0]["message"]["content"])
print(f"\n토큰: prompt={resp['usage']['prompt_tokens']}, completion={resp['usage']['completion_tokens']}")
