import requests
import base64
import json
import os

def test_demo():
    url = "https://voice-detector.up.railway.app/api/voice-detection"
    api_key = "sk_test_123456789"
    file_path = "sample voice 1.mp3"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return

    with open(file_path, "rb") as f:
        audio_content = f.read()
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')

    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }

    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    try:
        result = response.json()
        print("Response JSON:")
        print(json.dumps(result, indent=2))
    except:
        print("Response Content:")
        print(response.text)

if __name__ == "__main__":
    test_demo()
