
import io
import base64
import librosa
import numpy as np

# A truncated snippet of the C2PA manifest audio provided by the user
# This allows us to test if librosa/ffmpeg crashes on this specific metadata header
PARTIAL_B64 = "SUQzBAAAAAECBVRTU0UAAAAOAAADTGF2ZjYwLjE2LjEwMEdFT0IAAQFjAAADYXBwbGljYXRpb24veC1jMnBhLW1hbmlmZXN0LXN0b3JlAGMycGEAYzJwYSBtYW5pZmVzdCBzdG9yZQAAAECnanVtYgAAAB5qdW1kYzJwYQARABCAAACqADibcQNjMnBhAAAAQIFqdW1iAAAAR2p1bWRjMm1hABEAEIAAAKoAOJtxA3VybjpjMnBhOjkxNTY2OTZkLTVlNjktNDA3ZS1iY2FiLWYwNzYwZGY4OThiOAAAAAJ7anVtYgAAAClqdW1kYzJhcwARABCAAACqADibcQNjMnBhLmFzc2VydGlvbnMAAAAAy2p1bWIAAAApanVtZGNib3IAEQAQgAAAqgA4m3EDYzJwYS5hY3Rpb25zLnYyAAAAAJpjYm9yoWdhY3Rpb25zgaNmYWN0aW9ubGMycGEuY3JlYXRpdmV"

def test_load():
    print("Testing truncated C2PA/ID3 header load...")
    try:
        # Pad base64 if needed
        b64 = PARTIAL_B64 + "=" * (-len(PARTIAL_B64) % 4)
        data = base64.b64decode(b64)
        print(f"Decoded {len(data)} bytes")
        
        # Try loading
        f = io.BytesIO(data)
        try:
            # We expect this to FAIL because it's truncated, but we want to ensure it raises Python Exception
            # and NOT a Segfault/Crash
            librosa.load(f, sr=22050, duration=5)
            print("Loaded successfully (Unexpected for truncated)")
        except Exception as e:
            print(f"Caught expected exception: {e}")
            print("Test PASSED: Graceful failure")
            
    except Exception as e:
        print(f"Wrapper exception: {e}")

if __name__ == "__main__":
    test_load()
