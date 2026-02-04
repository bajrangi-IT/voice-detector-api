# Voice Detection API

A robust REST API for detecting AI-generated vs. Human voices using multi-language acoustic feature analysis.

## Features
- Supporting 5 major languages: Tamil, English, Hindi, Malayalam, Telugu.
- Dynamic acoustic feature extraction (Spectral Centroid, Flux, Pitch Jitter, Stability).
- Secure API key protection.
- High-performance processing (sub-500ms response).

## Tech Stack
- **Framework**: Flask
- **Acoustic Analysis**: Librosa, SciPy
- **Signal Processing**: NumPy

## API Endpoints

### 1. Root Check
`GET /`
- Returns API status.

### 2. Voice Detection
`POST /api/voice-detection`
- **Headers**: 
    - `x-api-key`: Your API Key
    - `Content-Type`: application/json
- **Body**:
    ```json
    {
      "language": "English",
      "audioFormat": "mp3",
      "audioBase64": "..."
    }
    ```
- **Response**:
    ```json
    {
      "status": "success",
      "language": "English",
      "classification": "AI_GENERATED",
      "confidenceScore": 0.92,
      "explanation": "..."
    }
    ```

### 3. Health Check
`GET /api/health`
- Returns system uptime and timestamp.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   python app.py
   ```

## License
MIT
