import requests
import base64
import json


with open("test.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

url = "http://127.0.0.1:8000/api/voice-detection"

headers = {
    "x-api-key": "sk_guvi_hackathon_2026",
    "Content-Type": "application/json"
}

payload = {
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": audio_base64
}

response = requests.post(url, headers=headers, data=json.dumps(payload))

print("Status Code:", response.status_code)
print("Response:", response.json())
