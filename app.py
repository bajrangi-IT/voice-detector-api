#!/usr/bin/env python3
"""
Voice Detection API - Detects AI-generated vs Human voices
Supports: Tamil, English, Hindi, Malayalam, Telugu
Author: Voice Detection Team
"""

import os
import re
import gc
import io
import base64
import binascii
import logging
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import librosa with error handling (non-fatal)
try:
    import librosa
    LIBROSA_AVAILABLE = True
    LIBROSA_ERROR = None
except Exception as e:
    logger.critical(f"Failed to import librosa: {e}")
    LIBROSA_AVAILABLE = False
    LIBROSA_ERROR = str(e)

import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Be tolerant with trailing slashes to avoid 405s from slash variations
app.url_map.strict_slashes = False

# Enable CORS for API routes and allow common headers/methods used by clients
CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=True,
    allow_headers=["Content-Type", "x-api-key", "Authorization"],
    methods=["GET", "POST", "OPTIONS", "HEAD"]
)

# Configuration: static keys for quick testing; prefer env var in production
API_KEYS = {
    'sk_test_123456789': 'test_user',
    'sk_prod_87654321': 'prod_user'
}
# Optionally add single API key from environment for convenience
ENV_API_KEY = os.environ.get("API_KEY")
if ENV_API_KEY:
    API_KEYS.setdefault(ENV_API_KEY, "env_user")

LANGUAGES = ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu']

# Track requests for simple stats
request_history = []


def check_api_key_value(key):
    """Return True if key is valid (case-sensitive string match)."""
    return key in API_KEYS


def extract_api_key_from_headers():
    """Look for x-api-key or Authorization: Bearer ... (case-insensitive)."""
    header_key = request.headers.get("x-api-key") or request.headers.get("X-API-KEY")
    if header_key:
        return header_key
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None


def validate_request(data):
    """Validate incoming request and accept common key variants."""
    if not data:
        return False, "No JSON data provided"

    language = (data.get('language') or data.get('Language') or '').strip().title()
    audio_format = (data.get('audioFormat') or data.get('format') or data.get('audioFormat') or '').lower().lstrip('.')
    audio_base64 = data.get('audioBase64') or data.get('audio_base64') or data.get('audio')

    if not language:
        return False, "Missing language"
    if language not in LANGUAGES:
        return False, f"Language '{language}' not supported"

    if not audio_format:
        return False, "Missing audio format"
    if audio_format != 'mp3':
        return False, "Only MP3 format is supported"

    if not audio_base64:
        return False, "No audio data provided"

    # Log length/prefix for debugging (do not log entire audio)
    try:
        logger.info("Received audioBase64 length=%d prefix=%s", len(audio_base64), (audio_base64[:40] + '...') if len(audio_base64) > 40 else audio_base64)
    except Exception:
        logger.info("Received audioBase64 (length unknown)")

    return True, (language, audio_format, audio_base64)


def decode_audio(audio_base64):
    """Decode base64 audio to bytes (tolerant).

    Handles data:audio/*;base64, whitespace, urlsafe base64, and returns clear errors.
    Returns (bytes, None) on success or (None, error_message) on failure.
    """
    try:
        if not isinstance(audio_base64, str) or not audio_base64:
            return None, "Audio payload is empty or not a string"

        # Strip common data URI prefix: data:audio/mp3;base64,AAAA...
        prefix_match = re.match(r'^\s*data:audio\/[a-z0-9.+-]+;base64,', audio_base64, flags=re.I)
        if prefix_match:
            audio_base64 = audio_base64[prefix_match.end():]

        # Remove whitespace/newlines
        audio_base64 = ''.join(audio_base64.split())

        # Sanity check length
        if len(audio_base64) < 16:
            return None, "Audio data too short or truncated"

        # Try strict/base64 decode (validate=True raises on non-base64 chars)
        try:
            audio_bytes = base64.b64decode(audio_base64, validate=True)
            if not audio_bytes:
                return None, "Decoded audio is empty"
            return audio_bytes, None
        except (binascii.Error, ValueError):
            # Try urlsafe variant (some clients use - and _)
            try:
                padding = '=' * (-len(audio_base64) % 4)
                audio_bytes = base64.urlsafe_b64decode(audio_base64 + padding)
                if not audio_bytes:
                    return None, "Decoded audio is empty after urlsafe decode"
                return audio_bytes, None
            except Exception as e:
                return None, f"Base64 decode failed: {str(e)}"

    except Exception as e:
        logger.exception("Unexpected error in decode_audio")
        return None, f"Exception while decoding audio: {e}"


def load_audio(audio_bytes):
    """Load audio with librosa (if available). Returns audio ndarray and sample rate."""
    if not LIBROSA_AVAILABLE:
        return None, None, f"Librosa not available: {LIBROSA_ERROR}"

    try:
        audio_file = io.BytesIO(audio_bytes)
        # librosa can accept a file-like object
        audio, sr = librosa.load(audio_file, sr=22050, duration=30)

        if len(audio) < sr * 0.3:  # Less than 0.3 seconds
            return None, None, "Audio too short"

        return audio, sr, None
    except Exception as e:
        logger.error(f"Audio loading error: {e}")
        return None, None, "Failed to load audio"


# Feature extraction and classifier classes (kept from your implementation)
class FeatureExtractor:
    """Extract acoustic features from audio"""
    def __init__(self, audio, sr):
        self.audio = audio
        self.sr = sr
        self.features = {}

    def normalize(self):
        max_val = np.max(np.abs(self.audio))
        if max_val > 1e-6:
            self.audio = self.audio / max_val

    def extract_pitch_features(self):
        if not LIBROSA_AVAILABLE:
            return {'pitch_consistency': 0.0, 'pitch_jitter': 0.0}

        try:
            f0 = librosa.yin(self.audio, fmin=50, fmax=500, trough_threshold=0.1)
            f0_valid = f0[f0 > 0]

            if len(f0_valid) < 5:
                return {'pitch_consistency': 0.0, 'pitch_jitter': 0.0}

            f0_norm = f0_valid / np.mean(f0_valid)
            pitch_consistency = 1.0 - np.std(f0_norm)
            pitch_consistency = np.clip(pitch_consistency, 0, 1)

            pitch_jitter = np.mean(np.abs(np.diff(f0_valid))) / np.mean(f0_valid)

            return {'pitch_consistency': float(pitch_consistency), 'pitch_jitter': float(pitch_jitter)}
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            return {'pitch_consistency': 0.0, 'pitch_jitter': 0.0}

    def extract_spectral_features(self):
        if not LIBROSA_AVAILABLE:
            return {'spectral_centroid': 0.0, 'spectral_flux': 0.0, 'spectral_contrast': 0.0}

        try:
            S = librosa.feature.melspectrogram(y=self.audio, sr=self.sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            centroid = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr)[0]
            contrast = librosa.feature.spectral_contrast(y=self.audio, sr=self.sr)
            flux = np.sqrt(np.sum(np.diff(S_db, axis=1)**2, axis=0))

            return {
                'spectral_centroid': float(np.mean(centroid)),
                'spectral_flux': float(np.mean(flux)),
                'spectral_contrast': float(np.mean(contrast))
            }
        except Exception as e:
            logger.warning(f"Spectral extraction failed: {e}")
            return {'spectral_centroid': 0.0, 'spectral_flux': 0.0, 'spectral_contrast': 0.0}

    def extract_mfcc_features(self):
        if not LIBROSA_AVAILABLE:
            return {'mfcc_variance': 0.0}

        try:
            mfcc = librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=13)
            mfcc_variance = float(np.mean(np.var(mfcc, axis=1)))
            return {'mfcc_variance': mfcc_variance}
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            return {'mfcc_variance': 0.0}

    def extract_temporal_features(self):
        if not LIBROSA_AVAILABLE:
            return {'onset_regularity': 0.0, 'zero_crossing_rate': 0.0}

        try:
            onset_env = librosa.onset.onset_strength(y=self.audio, sr=self.sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, units='frames')
            frame_times = librosa.frames_to_time(onset_frames, sr=self.sr)
            intervals = np.diff(frame_times) if len(frame_times) > 1 else np.array([0.1])

            if len(intervals) > 0 and np.mean(intervals) > 0:
                regularity = 1 - np.clip(np.std(intervals) / np.mean(intervals), 0, 1)
            else:
                regularity = 0.0

            zcr = np.mean(librosa.feature.zero_crossing_rate(self.audio))

            return {'onset_regularity': float(regularity), 'zero_crossing_rate': float(zcr)}
        except Exception as e:
            logger.warning(f"Temporal extraction failed: {e}")
            return {'onset_regularity': 0.0, 'zero_crossing_rate': 0.0}

    def extract_energy_features(self):
        if not LIBROSA_AVAILABLE:
            return {'energy_ratio': 0.0}

        try:
            stft = np.abs(librosa.stft(self.audio))
            power = np.abs(stft) ** 2
            energy = np.sum(power, axis=0)
            energy_ratio = np.std(energy) / (np.mean(energy) + 1e-6)
            return {'energy_ratio': float(energy_ratio)}
        except Exception as e:
            logger.warning(f"Energy extraction failed: {e}")
            return {'energy_ratio': 0.0}

    def extract_all(self):
        self.normalize()
        features = {}
        features.update(self.extract_pitch_features())
        features.update(self.extract_spectral_features())
        features.update(self.extract_mfcc_features())
        features.update(self.extract_temporal_features())
        features.update(self.extract_energy_features())
        return features


class VoiceClassifier:
    """Classify voice as AI or Human"""
    def __init__(self, features):
        self.features = features

    def score_ai_likelihood(self):
        score = 0.0

        pitch_cons = self.features.get('pitch_consistency', 0.5)
        if pitch_cons > 0.88:
            score += 4.5
        elif pitch_cons < 0.65:
            score -= 2.0

        jitter = self.features.get('pitch_jitter', 0.03)
        if jitter > 0.038:
            score += 5.5
        elif jitter < 0.008:
            score += 3.5

        flux = self.features.get('spectral_flux', 0.15)
        if flux > 0.18:
            score += 5.5
        elif flux < 0.12:
            score += 3.0

        mfcc_var = self.features.get('mfcc_variance', 0.5)
        if mfcc_var < 0.38:
            score += 4.0
        elif mfcc_var > 0.60:
            score -= 2.5

        onset_reg = self.features.get('onset_regularity', 0.5)
        if onset_reg > 0.82:
            score += 4.0
        elif onset_reg < 0.65:
            score -= 2.5

        energy = self.features.get('energy_ratio', 0.4)
        if energy < 0.25:
            score += 3.0

        zcr = self.features.get('zero_crossing_rate', 0.1)
        if zcr > 0.13:
            score += 2.5

        return score

    def classify(self):
        ai_score = self.score_ai_likelihood()
        confidence = 1.0 / (1.0 + np.exp(-ai_score / 3.0))
        confidence = np.clip(confidence, 0.01, 0.99)
        classification = 'AI_GENERATED' if confidence >= 0.50 else 'HUMAN'
        explanation = self.get_explanation(confidence)
        return {'classification': classification, 'confidence': confidence, 'explanation': explanation}

    def get_explanation(self, confidence):
        if confidence > 0.88:
            return "Unnatural pitch consistency and robotic speech patterns detected"
        elif confidence > 0.70:
            return "AI voice characteristics detected: artificial voice signatures and spectral artifacts"
        elif confidence > 0.50:
            return "Likely AI-generated voice with subtle artificial characteristics"
        elif confidence < 0.30:
            return "Natural human speech with high complexity and organic variation"
        else:
            return "Voice appears human with typical natural speech dynamics"


# Logging helper: show incoming request info to help debug intermittent 405/400
@app.before_request
def log_request_info():
    try:
        logger.info("Incoming request: %s %s", request.method, request.path)
        logger.debug("Headers: %s", dict(request.headers))
    except Exception:
        pass


@app.route('/', methods=['GET'])
def index():
    """Root route for verification"""
    return jsonify({
        'status': 'online',
        'message': 'Voice Detection API is running',
        'librosa_available': LIBROSA_AVAILABLE,
        'endpoints': {
            'detection': '/api/voice-detection',
            'health': '/api/health',
            'stats': '/api/stats'
        }
    }), 200


# Accept common verbs plus OPTIONS to prevent 405s from preflight or simple GET checks
@app.route('/api/voice-detection', methods=['GET', 'POST', 'OPTIONS', 'HEAD'])
@app.route('/api/voice-detection/', methods=['GET', 'POST', 'OPTIONS', 'HEAD'])
def detect_voice():
    """Main endpoint for voice detection"""
    # Respond to preflight quickly
    if request.method == 'OPTIONS':
        return ('', 204)

    # Allow a friendly GET/HEAD response for debugging clients
    if request.method in ('GET', 'HEAD'):
        return jsonify({
            'status': 'ok',
            'message': 'Send POST with JSON: {language, audioFormat, audioBase64} and header x-api-key'
        }), 200

    # Extract API key and validate
    api_key = extract_api_key_from_headers()
    if not api_key or not check_api_key_value(api_key):
        logger.warning(f"Invalid API key attempt: {api_key}")
        return jsonify({'status': 'error', 'message': 'Invalid API key or malformed request'}), 401

    # Parse JSON safely
    data = request.get_json(silent=True)
    is_valid, result = validate_request(data)
    if not is_valid:
        logger.warning(f"Invalid request: {result}")
        return jsonify({'status': 'error', 'message': result}), 400

    language, audio_format, audio_base64 = result

    # Decode audio
    audio_bytes, decode_error = decode_audio(audio_base64)

    # Aggressive memory cleanup for large base64 blobs
    try:
        del audio_base64, data, result
    except Exception:
        pass
    gc.collect()

    if decode_error:
        logger.warning(f"Decode error: {decode_error}")
        return jsonify({'status': 'error', 'message': decode_error}), 400

    # Load audio into waveform
    audio, sr, load_error = load_audio(audio_bytes)

    # Free raw bytes
    try:
        del audio_bytes
    except Exception:
        pass
    gc.collect()

    if load_error:
        logger.warning(f"Load error: {load_error}")
        return jsonify({'status': 'error', 'message': load_error}), 400

    # Extract features
    try:
        extractor = FeatureExtractor(audio, sr)
        features = extractor.extract_all()
        logger.info(f"Features extracted for {language}")
        # free large objects promptly
        del audio, extractor
        gc.collect()
    except Exception as e:
        logger.exception("Feature extraction error")
        return jsonify({'status': 'error', 'message': 'Feature extraction failed'}), 500

    # Classify
    try:
        classifier = VoiceClassifier(features)
        result = classifier.classify()
        logger.info(f"Classification: {result['classification']} (confidence: {result['confidence']:.2f})")
    except Exception as e:
        logger.exception("Classification error")
        return jsonify({'status': 'error', 'message': 'Classification failed'}), 500

    response = {
        'status': 'success',
        'language': language,
        'classification': result['classification'],
        'confidenceScore': round(result['confidence'], 2),
        'explanation': result['explanation']
    }

    # Save simple request log
    request_history.append({
        'timestamp': datetime.now().isoformat(),
        'language': language,
        'classification': result['classification']
    })

    return jsonify(response), 200


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'supported_languages': LANGUAGES
    }), 200


@app.route('/api/stats', methods=['GET'])
def stats():
    api_key = extract_api_key_from_headers()
    if not api_key or not check_api_key_value(api_key):
        return jsonify({'status': 'error', 'message': 'Invalid API key or malformed request'}), 401

    total = len(request_history)
    ai_count = sum(1 for req in request_history if req['classification'] == 'AI_GENERATED')
    human_count = total - ai_count

    return jsonify({
        'total_requests': total,
        'ai_generated_count': ai_count,
        'human_count': human_count
    }), 200


# Error handlers
@app.errorhandler(400)
def bad_request(error):
    logger.warning(f"400 for {request.method} {request.path} - {error}")
    return jsonify({'status': 'error', 'message': 'Bad Request: Malformed JSON or missing data'}), 400


@app.errorhandler(401)
def unauthorized(error):
    logger.warning(f"401 for {request.method} {request.path} - {error}")
    return jsonify({'status': 'error', 'message': 'Unauthorized: Invalid API key'}), 401


@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 for {request.method} {request.path} - {error}")
    return jsonify({'status': 'error', 'message': 'Resource not found. Use /api/voice-detection or /api/health'}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    logger.warning(f"405 Method Not Allowed: {request.method} {request.path} - Headers: {dict(request.headers)}")
    return jsonify({'status': 'error', 'message': 'Method not allowed. Use POST to /api/voice-detection'}), 405


@app.errorhandler(500)
def server_error(error):
    logger.error(f"Internal Server Error: {error}")
    return jsonify({'status': 'error', 'message': 'Internal Server Error. Please check logs.'}), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Voice Detection API - Starting...")
    print("="*80)
    print("Supported Languages: Tamil, English, Hindi, Malayalam, Telugu")
    print("Endpoint: http://0.0.0.0:5000/api/voice-detection")
    print("="*80 + "\n")

    # In production, Gunicorn (Procfile) should be used. This is for local debug.
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), threaded=True)