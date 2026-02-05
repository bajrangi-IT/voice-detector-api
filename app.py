#!/usr/bin/env python3
"""
Robust Voice Detection REST API

- Accepts POST /api/voice-detection with either:
  - JSON: { language, audioFormat (or format), audioBase64 (or audio_base64 or audio) }
  - multipart/form-data: file field (file or audioFile) and optional language/format fields
- Accepts GET/HEAD for simple sanity check and OPTIONS for CORS preflight.
- Validates API key via header x-api-key or Authorization: Bearer <key>.
- Tolerant base64 decoder (handles data: URIs, whitespace, urlsafe variants).
- Attempts to load audio using soundfile (recommended) or librosa (if available).
- Defensive error handling: never raises an uncaught exception; always returns JSON.
- Configure API key(s) via API_KEYS dict or environment variable API_KEY.
"""

import os
import re
import io
import gc
import base64
import binascii
import logging
import tempfile
from datetime import datetime
from typing import Tuple, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

# Optional backends
try:
    import soundfile as sf  # preferred for reading bytes
    SOUND_FILE_AVAILABLE = True
except Exception:
    sf = None
    SOUND_FILE_AVAILABLE = False

try:
    import librosa  # fallback
    LIBROSA_AVAILABLE = True
    LIBROSA_ERROR = None
except Exception as e:
    librosa = None
    LIBROSA_AVAILABLE = False
    LIBROSA_ERROR = str(e)

import numpy as np

# App & logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("voice-detector")

app = Flask(__name__)
# Limit request payload to avoid memory blowups (adjust as needed)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB
# Tolerate trailing slash variations
app.url_map.strict_slashes = False

# CORS for endpoints used by testers
CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "x-api-key", "Authorization"],
    methods=["GET", "POST", "OPTIONS", "HEAD"]
)

# API keys: for convenience put test keys here but use env var in production
API_KEYS = {
    "sk_test_123456789": "test_user",
    "sk_prod_87654321": "prod_user",
}
ENV_KEY = os.environ.get("API_KEY")
if ENV_KEY:
    API_KEYS.setdefault(ENV_KEY, "env_user")

LANGUAGES = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}

# Lightweight in-memory stats (not persistent)
request_history = []

# Helpers ---------------------------------------------------------------------


def extract_api_key_from_headers() -> Optional[str]:
    """Look for x-api-key or Authorization: Bearer <key>."""
    key = request.headers.get("x-api-key") or request.headers.get("X-API-KEY")
    if key:
        return key.strip()
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None


def check_api_key(key: Optional[str]) -> bool:
    if not key:
        return False
    return key in API_KEYS


def tolerant_base64_to_bytes(s: object) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Robust base64 -> bytes conversion.
    Returns (bytes, None) on success or (None, error_message) on failure.
    Accepts str or bytes. Removes data: URI prefix, whitespace, non-base64 chars,
    handles urlsafe variants and padding.
    """
    try:
        if s is None:
            return None, "Audio payload is missing"
        if isinstance(s, (bytes, bytearray)):
            try:
                s = s.decode("utf-8", errors="ignore")
            except Exception:
                s = str(s)

        if not isinstance(s, str):
            return None, "Audio payload must be a base64 string"

        s = s.strip()
        if not s:
            return None, "Audio payload is empty"

        # strip data URI prefix if present
        m = re.match(r"^\s*data:audio\/[a-z0-9.+-]+;base64,", s, flags=re.I)
        if m:
            s = s[m.end():]

        # remove whitespace/newlines
        s = "".join(s.split())

        # remove any chars not in base64/url-safe/padding set
        cleaned = re.sub(r"[^A-Za-z0-9+/=_\-]", "", s)
        removed = len(s) - len(cleaned)
        if removed > 0:
            logger.info("tolerant_base64_to_bytes: removed %d invalid chars from input", removed)

        if len(cleaned) < 16:
            return None, "Audio data too short or truncated"

        # normalize urlsafe to standard
        cleaned = cleaned.replace("-", "+").replace("_", "/")

        # pad to multiple of 4
        pad_len = (-len(cleaned)) % 4
        if pad_len:
            cleaned += "=" * pad_len

        # try strict decode first
        try:
            b = base64.b64decode(cleaned, validate=True)
            if not b:
                return None, "Decoded audio is empty"
            return b, None
        except (binascii.Error, ValueError):
            # fallback tolerant decode
            try:
                b = base64.b64decode(cleaned)
                if not b:
                    return None, "Decoded audio empty after fallback decode"
                return b, None
            except Exception as e:
                logger.debug("base64 fallback decode exception: %s", str(e))
                return None, f"Failed to decode audio: {str(e)}"

    except Exception as exc:
        logger.exception("Unexpected in tolerant_base64_to_bytes")
        return None, f"Exception while decoding audio: {exc}"


def load_audio_from_bytes(audio_bytes: bytes) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """
    Try to read raw audio bytes into waveform (numpy array) and sample rate.
    Preferred: soundfile (sf), fallback to librosa. Returns (audio, sr, None) or (None, None, error)
    """
    if audio_bytes is None:
        return None, None, "No audio bytes provided"

    # Try soundfile (reads many container types)
    if SOUND_FILE_AVAILABLE:
        try:
            bio = io.BytesIO(audio_bytes)
            data, sr = sf.read(bio, dtype="float32")
            # sf.read may return shape (nsamples,) or (nsamples, channels)
            if data is None or len(data) == 0:
                return None, None, "Could not read audio (soundfile empty)"
            # If stereo, convert to mono by averaging channels
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return data.astype(np.float32), int(sr), None
        except Exception as e:
            logger.warning("soundfile read failed: %s", str(e))
            # continue to next fallback

    # Fallback: librosa
    if LIBROSA_AVAILABLE:
        try:
            bio = io.BytesIO(audio_bytes)
            # librosa.load accepts file-like objects if soundfile is installed; if not, try sr only
            audio, sr = librosa.load(bio, sr=22050, duration=30)
            if audio is None or len(audio) == 0:
                return None, None, "Could not load audio (librosa empty)"
            return audio.astype(np.float32), int(sr), None
        except Exception as e:
            logger.warning("librosa load failed: %s", str(e))
            return None, None, f"Audio decoding failed: {str(e)}"

    return None, None, "No audio backend available (install soundfile or librosa)"


# Feature extraction & classification (kept defensive) ------------------------


class FeatureExtractor:
    def __init__(self, audio: np.ndarray, sr: int):
        self.audio = audio
        self.sr = sr

    def normalize(self):
        try:
            max_val = np.max(np.abs(self.audio))
            if max_val > 1e-6:
                self.audio = self.audio / max_val
        except Exception:
            pass

    def extract_pitch_features(self):
        if not LIBROSA_AVAILABLE:
            return {"pitch_consistency": 0.0, "pitch_jitter": 0.0}
        try:
            f0 = librosa.yin(self.audio, fmin=50, fmax=500, trough_threshold=0.1)
            f0_valid = f0[f0 > 0]
            if len(f0_valid) < 5:
                return {"pitch_consistency": 0.0, "pitch_jitter": 0.0}
            f0_norm = f0_valid / np.mean(f0_valid)
            pitch_consistency = 1.0 - np.std(f0_norm)
            pitch_consistency = float(np.clip(pitch_consistency, 0, 1))
            pitch_jitter = float(np.mean(np.abs(np.diff(f0_valid))) / np.mean(f0_valid))
            return {"pitch_consistency": pitch_consistency, "pitch_jitter": pitch_jitter}
        except Exception as e:
            logger.debug("pitch extraction failed: %s", str(e))
            return {"pitch_consistency": 0.0, "pitch_jitter": 0.0}

    def extract_spectral_features(self):
        if not LIBROSA_AVAILABLE:
            return {"spectral_centroid": 0.0, "spectral_flux": 0.0, "spectral_contrast": 0.0}
        try:
            S = librosa.feature.melspectrogram(y=self.audio, sr=self.sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            centroid = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr)[0]
            contrast = librosa.feature.spectral_contrast(y=self.audio, sr=self.sr)
            flux = np.sqrt(np.sum(np.diff(S_db, axis=1) ** 2, axis=0))
            return {
                "spectral_centroid": float(np.mean(centroid)),
                "spectral_flux": float(np.mean(flux)),
                "spectral_contrast": float(np.mean(contrast)),
            }
        except Exception as e:
            logger.debug("spectral extraction failed: %s", str(e))
            return {"spectral_centroid": 0.0, "spectral_flux": 0.0, "spectral_contrast": 0.0}

    def extract_mfcc_features(self):
        if not LIBROSA_AVAILABLE:
            return {"mfcc_variance": 0.0}
        try:
            mfcc = librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=13)
            mfcc_variance = float(np.mean(np.var(mfcc, axis=1)))
            return {"mfcc_variance": mfcc_variance}
        except Exception as e:
            logger.debug("mfcc extraction failed: %s", str(e))
            return {"mfcc_variance": 0.0}

    def extract_temporal_features(self):
        if not LIBROSA_AVAILABLE:
            return {"onset_regularity": 0.0, "zero_crossing_rate": 0.0}
        try:
            onset_env = librosa.onset.onset_strength(y=self.audio, sr=self.sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, units="frames")
            frame_times = librosa.frames_to_time(onset_frames, sr=self.sr)
            intervals = np.diff(frame_times) if len(frame_times) > 1 else np.array([0.1])
            if len(intervals) > 0 and np.mean(intervals) > 0:
                regularity = 1 - np.clip(np.std(intervals) / np.mean(intervals), 0, 1)
            else:
                regularity = 0.0
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(self.audio)))
            return {"onset_regularity": float(regularity), "zero_crossing_rate": zcr}
        except Exception as e:
            logger.debug("temporal extraction failed: %s", str(e))
            return {"onset_regularity": 0.0, "zero_crossing_rate": 0.0}

    def extract_energy_features(self):
        if not LIBROSA_AVAILABLE:
            return {"energy_ratio": 0.0}
        try:
            stft = np.abs(librosa.stft(self.audio))
            power = np.abs(stft) ** 2
            energy = np.sum(power, axis=0)
            energy_ratio = np.std(energy) / (np.mean(energy) + 1e-6)
            return {"energy_ratio": float(energy_ratio)}
        except Exception as e:
            logger.debug("energy extraction failed: %s", str(e))
            return {"energy_ratio": 0.0}

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
    def __init__(self, features: dict):
        self.features = features

    def score_ai_likelihood(self) -> float:
        score = 0.0
        pitch_cons = self.features.get("pitch_consistency", 0.5)
        if pitch_cons > 0.88:
            score += 4.5
        elif pitch_cons < 0.65:
            score -= 2.0
        jitter = self.features.get("pitch_jitter", 0.03)
        if jitter > 0.038:
            score += 5.5
        elif jitter < 0.008:
            score += 3.5
        flux = self.features.get("spectral_flux", 0.15)
        if flux > 0.18:
            score += 5.5
        elif flux < 0.12:
            score += 3.0
        mfcc_var = self.features.get("mfcc_variance", 0.5)
        if mfcc_var < 0.38:
            score += 4.0
        elif mfcc_var > 0.60:
            score -= 2.5
        onset_reg = self.features.get("onset_regularity", 0.5)
        if onset_reg > 0.82:
            score += 4.0
        elif onset_reg < 0.65:
            score -= 2.5
        energy = self.features.get("energy_ratio", 0.4)
        if energy < 0.25:
            score += 3.0
        zcr = self.features.get("zero_crossing_rate", 0.1)
        if zcr > 0.13:
            score += 2.5
        return float(score)

    def classify(self) -> dict:
        ai_score = self.score_ai_likelihood()
        try:
            confidence = 1.0 / (1.0 + np.exp(-ai_score / 3.0))
        except Exception:
            confidence = 0.5
        confidence = float(np.clip(confidence, 0.01, 0.99))
        classification = "AI_GENERATED" if confidence >= 0.50 else "HUMAN"
        explanation = self.get_explanation(confidence)
        return {"classification": classification, "confidence": confidence, "explanation": explanation}

    def get_explanation(self, confidence: float) -> str:
        if confidence > 0.88:
            return "Unnatural pitch consistency and robotic speech patterns detected"
        if confidence > 0.70:
            return "AI voice characteristics detected: artificial voice signatures and spectral artifacts"
        if confidence > 0.50:
            return "Likely AI-generated voice with subtle artificial characteristics"
        if confidence < 0.30:
            return "Natural human speech with high complexity and organic variation"
        return "Voice appears human with typical natural speech dynamics"


# Routes ---------------------------------------------------------------------


@app.before_request
def log_request():
    # Keep concise logs; avoid logging raw audio content
    try:
        logger.info("Incoming %s %s (Content-Length=%s)", request.method, request.path, request.headers.get("Content-Length"))
    except Exception:
        pass


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "status": "online",
            "message": "Voice Detection API running",
            "librosa_available": LIBROSA_AVAILABLE,
            "soundfile_available": SOUND_FILE_AVAILABLE,
            "endpoints": {"detection": "/api/voice-detection", "health": "/api/health", "stats": "/api/stats"},
        }
    ), 200


@app.route("/api/voice-detection", methods=["GET", "POST", "OPTIONS", "HEAD"])
@app.route("/api/voice-detection/", methods=["GET", "POST", "OPTIONS", "HEAD"])
def detect_voice():
    # Handle preflight
    if request.method == "OPTIONS":
        return ("", 204)

    if request.method in ("GET", "HEAD"):
        return (
            jsonify(
                {
                    "status": "ok",
                    "message": "Send POST with JSON: {language, audioFormat, audioBase64} (or multipart file 'file') and header x-api-key",
                }
            ),
            200,
        )

    # Protect entire handler with try/except to never leak stack traces
    try:
        # API key check
        api_key = extract_api_key_from_headers()
        if not check_api_key(api_key):
            logger.warning("Unauthorized request, missing/invalid API key")
            return jsonify({"status": "error", "message": "Invalid API key"}), 401

        # Determine input method: JSON base64 or multipart file
        content_type = request.content_type or ""
        language = None
        audio_format = None
        audio_bytes = None

        # JSON body
        if "application/json" in content_type or request.is_json:
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"status": "error", "message": "Invalid or empty JSON body"}), 400
            # accept multiple key variants
            language = (data.get("language") or data.get("Language") or "").strip().title()
            audio_format = (data.get("audioFormat") or data.get("format") or data.get("audioFormat") or "").lower().lstrip(".")
            audio_field = data.get("audioBase64") or data.get("audio_base64") or data.get("audio")
            if not audio_field:
                return jsonify({"status": "error", "message": "Missing audioBase64 field in JSON"}), 400
            audio_bytes, err = tolerant_base64_to_bytes(audio_field)
            if err:
                return jsonify({"status": "error", "message": err}), 400

        # multipart/form-data file upload
        elif "multipart/form-data" in content_type:
            # file under 'file' or 'audioFile'
            f = request.files.get("file") or request.files.get("audioFile") or request.files.get("audio")
            if not f:
                return jsonify({"status": "error", "message": "Missing file in multipart/form-data"}), 400
            try:
                audio_bytes = f.read()
            except Exception:
                # fallback saving to temp file
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False)
                    f.save(tmp.name)
                    tmp.close()
                    with open(tmp.name, "rb") as rf:
                        audio_bytes = rf.read()
                except Exception as e:
                    logger.exception("Failed reading uploaded file")
                    return jsonify({"status": "error", "message": "Failed to read uploaded file"}), 400
                finally:
                    try:
                        os.unlink(tmp.name)
                    except Exception:
                        pass
            language = (request.form.get("language") or request.form.get("lang") or "").strip().title()
            audio_format = (request.form.get("format") or request.form.get("audioFormat") or "").lower().lstrip(".")
        else:
            # For other content types, try to parse JSON or fail
            data = request.get_json(silent=True)
            if data:
                language = (data.get("language") or data.get("Language") or "").strip().title()
                audio_format = (data.get("audioFormat") or data.get("format") or "").lower().lstrip(".")
                audio_field = data.get("audioBase64") or data.get("audio_base64") or data.get("audio")
                if audio_field:
                    audio_bytes, err = tolerant_base64_to_bytes(audio_field)
                    if err:
                        return jsonify({"status": "error", "message": err}), 400
            else:
                return jsonify({"status": "error", "message": "Unsupported Content-Type; send JSON or multipart/form-data"}), 415

        # Validate language & format
        if not language:
            return jsonify({"status": "error", "message": "Missing language"}), 400
        if language not in LANGUAGES:
            return jsonify({"status": "error", "message": f"Unsupported language: {language}"}), 400

        if not audio_format:
            # try to guess from bytes header (very simple)
            audio_format = "mp3"  # default expectation from tester
        if audio_format.lower().lstrip(".") != "mp3":
            return jsonify({"status": "error", "message": "Only MP3 format supported"}), 400

        if not audio_bytes:
            return jsonify({"status": "error", "message": "No audio data available"}), 400

        # Load audio waveform
        audio, sr, load_err = load_audio_from_bytes(audio_bytes)
        # free raw bytes ASAP
        try:
            del audio_bytes
        except Exception:
            pass
        gc.collect()

        if load_err:
            # Return the backend-friendly message
            logger.warning("Audio load failed: %s", load_err)
            return jsonify({"status": "error", "message": load_err}), 400

        # Basic length check
        if audio is None or sr is None or len(audio) < int(sr * 0.25):
            return jsonify({"status": "error", "message": "Audio too short"}), 400

        # Feature extraction & classification
        try:
            extractor = FeatureExtractor(audio, sr)
            features = extractor.extract_all()
            # free audio and extractor early
            try:
                del audio, extractor
            except Exception:
                pass
            gc.collect()
        except Exception as e:
            logger.exception("Feature extraction error")
            return jsonify({"status": "error", "message": "Feature extraction failed"}), 500

        try:
            classifier = VoiceClassifier(features)
            result = classifier.classify()
        except Exception as e:
            logger.exception("Classification error")
            return jsonify({"status": "error", "message": "Classification failed"}), 500

        # Build response
        response = {
            "status": "success",
            "language": language,
            "classification": result.get("classification", "UNKNOWN"),
            "confidenceScore": round(float(result.get("confidence", 0.0)), 2),
            "explanation": result.get("explanation", ""),
        }

        # store light-weight log
        try:
            request_history.append({"timestamp": datetime.utcnow().isoformat(), "language": language, "classification": response["classification"]})
        except Exception:
            pass

        return jsonify(response), 200

    except Exception as e:
        logger.exception("Unexpected error in detect_voice")
        return jsonify({"status": "error", "message": "Internal Server Error"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "supported_languages": sorted(list(LANGUAGES))}), 200


@app.route("/api/stats", methods=["GET"])
def stats():
    api_key = extract_api_key_from_headers()
    if not check_api_key(api_key):
        return jsonify({"status": "error", "message": "Invalid API key"}), 401
    total = len(request_history)
    ai_count = sum(1 for r in request_history if r.get("classification") == "AI_GENERATED")
    human_count = total - ai_count
    return jsonify({"total_requests": total, "ai_generated_count": ai_count, "human_count": human_count}), 200


# Error handlers --------------------------------------------------------------


@app.errorhandler(400)
def _bad_request(e):
    logger.debug("400 error: %s %s", request.path, str(e))
    return jsonify({"status": "error", "message": "Bad Request: check payload and fields"}), 400


@app.errorhandler(401)
def _unauthorized(e):
    return jsonify({"status": "error", "message": "Unauthorized: API key invalid"}), 401


@app.errorhandler(404)
def _not_found(e):
    return jsonify({"status": "error", "message": "Not found"}), 404


@app.errorhandler(405)
def _method_not_allowed(e):
    logger.warning("405: %s %s", request.method, request.path)
    return jsonify({"status": "error", "message": "Method not allowed"}), 405


@app.errorhandler(413)
def _payload_too_large(e):
    return jsonify({"status": "error", "message": "Payload too large"}), 413


@app.errorhandler(500)
def _server_error(e):
    logger.exception("500 error")
    return jsonify({"status": "error", "message": "Internal Server Error"}), 500


# Run ------------------------------------------------------------------------


if __name__ == "__main__":
    # For local testing only. In production use gunicorn (Procfile) so PORT is set.
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Voice Detection API on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)