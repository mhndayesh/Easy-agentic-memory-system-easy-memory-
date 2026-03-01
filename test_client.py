import requests
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

HOST = config["server"]["host"]
if HOST == "0.0.0.0":
    HOST = "127.0.0.1"
PORT = config["server"]["port"]
PROXY_URL = f"http://{HOST}:{PORT}/v1/chat/completions"

def send_msg(msg):
    print(f"\nUser: {msg}")
    payload = {
        "messages": [{"role": "user", "content": msg}],
        "stream": False
    }
    try:
        response = requests.post(PROXY_URL, json=payload, timeout=240)
        if response.status_code == 200:
            res_json = response.json()
            message = res_json["choices"][0]["message"]
            content = message.get("content", "")
            print(f"Agent: {content}")
            if "tool_calls" in message:
                print(f"[TOOL CALLED]: {message['tool_calls']}")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        send_msg(" ".join(sys.argv[1:]))
    else:
        print("Usage: python test_client.py \"Your message here\"")
        print("Example: python test_client.py /save")
