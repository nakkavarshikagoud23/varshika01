import requests
import json

API_KEY = "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"

# 1. Verify the token is valid by checking whoami
print("=== Checking token validity ===")
r = requests.get("https://huggingface.co/api/whoami", headers={"Authorization": f"Bearer {API_KEY}"})
print(f"Status: {r.status_code}")
if r.status_code == 200:
    info = r.json()
    print(f"User: {info.get('name', 'N/A')}")
    print(f"Token type: {info.get('auth', {}).get('type', 'N/A')}")
    print(f"Token name: {info.get('auth', {}).get('accessToken', {}).get('displayName', 'N/A')}")
    
    # Check for fine-grained permissions
    fgp = info.get('auth', {}).get('accessToken', {}).get('fineGrained', {})
    print(f"Fine-grained permissions: {json.dumps(fgp, indent=2) if fgp else 'None (legacy token)'}")
    
    # Full auth info
    print(f"\nFull auth section: {json.dumps(info.get('auth', {}), indent=2)}")
else:
    print(f"Token invalid: {r.text[:200]}")
