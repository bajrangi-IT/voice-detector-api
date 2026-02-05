#!/usr/bin/env python3
"""
Voice Detection API - Improved Detector

Goals:
- More robust and efficient feature extraction.
- Ensemble-like scoring with multiple features and calibrated confidence.
- Detailed per-feature contributions and a reliability score.
- Same robust input handling as before (JSON base64 or multipart; ffmpeg fallback).
- Defensive programming to avoid crashes in production.

Notes:
- For best accuracy, deploy with soundfile/librosa + ffmpeg available (see previous instructions).
- This file extends the previous robust implementation with expanded feature set and
  an improved scoring & explanation mechanism.
"""

import os
import re
import io
import gc
import math
import base64
import binascii
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS

# Optional backends
try:
    import soundfile as sf  # preferred
    SOUND_FILE_AVAILABLE = True
except Exception:
    sf = None
    SOUND_FILE_AVAILABLE = False

try:
    import librosa  # fallback and feature extraction
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
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB max
app.url_map.strict_slashes = False

CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False,
     allow_headers=["Content-Type", "x-api-key", "Authorization"], methods=["GET", "POST", "OPTIONS", "HEAD"])

# API keys: keep test keys but allow override via env
API_KEYS = {
    "sk_test_123456789": "test_user",
    "sk_prod_87654321": "prod_user",
}
ENV_KEY = os.environ.get("API_KEY")
if ENV_KEY:
    API_KEYS.setdefault(ENV_KEY, "env_user")

LANGUAGES = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}

request_history = []


# -------------------- Utility & I/O helpers ---------------------------------


def extract_api_key_from_headers() -> Optional[str]:
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
    """Robust base64 to bytes with cleaning and urlsafe handling."""
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
        # strip data URI
        m = re.match(r"^\s*data:audio\/[a-z0-9.+-]+;base64,", s, flags=re.I)
        if m:
            s = s[m.end():]
        s = "".join(s.split())
        cleaned = re.sub(r"[^A-Za-z0-9+/=_\-]", "", s)
        removed = len(s) - len(cleaned)
        if removed > 0:
            logger.info("tolerant_base64_to_bytes: removed %d invalid chars from input", removed)
        if len(cleaned) < 16:
            return None, "Audio data too short or truncated"
        cleaned = cleaned.replace("-", "+").replace("_", "/")
        pad_len = (-len(cleaned)) % 4
        if pad_len:
            cleaned += "=" * pad_len
        try:
            b = base64.b64decode(cleaned, validate=True)
            if not b:
                return None, "Decoded audio is empty"
            return b, None
        except (binascii.Error, ValueError):
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


def ffmpeg_transcode_to_wav_bytes(in_bytes: bytes, timeout: int = 20) -> Tuple[Optional[bytes], Optional[str]]:
    """Use ffmpeg binary (if available) to transcode arbitrary audio bytes to WAV bytes."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None, "ffmpeg not available"
    try:
        proc = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "wav", "pipe:1"],
            input=in_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode("utf-8", errors="ignore")
            logger.debug("ffmpeg failed rc=%s err=%s", proc.returncode, err)
            return None, f"ffmpeg failed: {err.strip()[:200]}"
        return proc.stdout, None
    except subprocess.TimeoutExpired:
        return None, "ffmpeg timed out"
    except Exception as e:
        logger.exception("ffmpeg transcode error")
        return None, f"ffmpeg error: {e}"


def load_audio_from_bytes(audio_bytes: bytes) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """
    Read audio bytes to (audio ndarray mono float32, sample_rate).
    Strategy:
      1) soundfile (sf) if available
      2) librosa if available
      3) ffmpeg transcode -> read WAV via soundfile/librosa
    """
    if not audio_bytes:
        return None, None, "No audio bytes provided"

    # 1) soundfile
    if SOUND_FILE_AVAILABLE:
        try:
            bio = io.BytesIO(audio_bytes)
            data, sr = sf.read(bio, dtype="float32")
            if data is None or len(data) == 0:
                raise ValueError("soundfile returned empty")
            if getattr(data, "ndim", 1) > 1:
                data = np.mean(data, axis=1)
            return data.astype(np.float32), int(sr), None
        except Exception as e:
            logger.debug("soundfile read failed: %s", str(e))

    # 2) librosa
    if LIBROSA_AVAILABLE:
        try:
            bio = io.BytesIO(audio_bytes)
            audio, sr = librosa.load(bio, sr=22050, duration=60.0)
            if audio is None or len(audio) == 0:
                raise ValueError("librosa returned empty")
            return audio.astype(np.float32), int(sr), None
        except Exception as e:
            logger.debug("librosa load failed: %s", str(e))

    # 3) ffmpeg fallback
    wav_bytes, ff_err = ffmpeg_transcode_to_wav_bytes(audio_bytes)
    if ff_err:
        logger.debug("ffmpeg transcode not usable: %s", ff_err)
        return None, None, "Audio decoding failed: no backend could decode MP3/Audio (install ffmpeg or enable libsndfile/mp3 support)."

    # Try reading WAV bytes
    if SOUND_FILE_AVAILABLE:
        try:
            bio = io.BytesIO(wav_bytes)
            data, sr = sf.read(bio, dtype="float32")
            if data is None or len(data) == 0:
                raise ValueError("soundfile read of wav returned empty")
            if getattr(data, "ndim", 1) > 1:
                data = np.mean(data, axis=1)
            return data.astype(np.float32), int(sr), None
        except Exception as e:
            logger.debug("soundfile read of ffmpeg wav failed: %s", str(e))

    if LIBROSA_AVAILABLE:
        try:
            bio = io.BytesIO(wav_bytes)
            audio, sr = librosa.load(bio, sr=22050, duration=60.0)
            if audio is None or len(audio) == 0:
                raise ValueError("librosa read of wav returned empty")
            return audio.astype(np.float32), int(sr), None
        except Exception as e:
            logger.debug("librosa read of ffmpeg wav failed: %s", str(e))

    return None, None, "Audio decoding failed after ffmpeg transcode: no library could read WAV output."


# -------------------- Feature extraction (expanded) -------------------------


class FeatureExtractor:
    """
    Extended feature extractor. Attempts to compute a broad set of acoustic features
    that are informative for distinguishing AI-generated speech vs human speech.
    All methods are defensive: they return fallback values if specific extraction fails.
    """

    def __init__(self, audio: np.ndarray, sr: int):
        self.audio = np.copy(audio.astype(np.float32))
        self.sr = int(sr)
        self._frame_length = 2048
        self._hop_length = 512

    def normalize(self):
        try:
            maxv = np.max(np.abs(self.audio))
            if maxv > 1e-6:
                self.audio = self.audio / maxv
        except Exception:
            pass

    def rms_stats(self):
        try:
            rms = librosa.feature.rms(y=self.audio, frame_length=self._frame_length, hop_length=self._hop_length)[0]
            return {
                "rms_mean": float(np.mean(rms)),
                "rms_std": float(np.std(rms)),
                "rms_skew": float(np.mean(((rms - np.mean(rms)) ** 3))) if len(rms) > 2 else 0.0,
            }
        except Exception:
            return {"rms_mean": 0.0, "rms_std": 0.0, "rms_skew": 0.0}

    def spectral_stats(self):
        try:
            centroid = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr, n_fft=self._frame_length, hop_length=self._hop_length)[0]
            rolloff = librosa.feature.spectral_rolloff(y=self.audio, sr=self.sr, n_fft=self._frame_length, hop_length=self._hop_length)[0]
            flux = np.sqrt(np.sum(np.diff(librosa.power_to_db(librosa.feature.melspectrogram(y=self.audio, sr=self.sr, n_mels=64)), axis=1) ** 2, axis=0))
            return {
                "spectral_centroid_mean": float(np.mean(centroid)),
                "spectral_rolloff_mean": float(np.mean(rolloff)),
                "spectral_flux_mean": float(np.mean(flux)) if len(flux) > 0 else 0.0,
            }
        except Exception:
            return {"spectral_centroid_mean": 0.0, "spectral_rolloff_mean": 0.0, "spectral_flux_mean": 0.0}

    def mfcc_stats(self):
        try:
            mfcc = librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=13, hop_length=self._hop_length)
            return {"mfcc_var_mean": float(np.mean(np.var(mfcc, axis=1)))}
        except Exception:
            return {"mfcc_var_mean": 0.0}

    def pitch_stats(self):
        try:
            f0 = librosa.yin(self.audio, fmin=50, fmax=500, frame_length=self._frame_length, hop_length=self._hop_length, trough_threshold=0.1)
            f0_valid = f0[f0 > 0]
            if len(f0_valid) < 3:
                return {"pitch_consistency": 0.0, "pitch_jitter": 0.0, "median_f0": 0.0}
            f0_norm = f0_valid / np.mean(f0_valid)
            pitch_consistency = float(np.clip(1.0 - np.std(f0_norm), 0.0, 1.0))
            pitch_jitter = float(np.mean(np.abs(np.diff(f0_valid))) / (np.mean(f0_valid) + 1e-9))
            return {"pitch_consistency": pitch_consistency, "pitch_jitter": pitch_jitter, "median_f0": float(np.median(f0_valid))}
        except Exception:
            return {"pitch_consistency": 0.0, "pitch_jitter": 0.0, "median_f0": 0.0}

    def temporal_stats(self):
        try:
            onset_env = librosa.onset.onset_strength(y=self.audio, sr=self.sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, units="frames")
            frame_times = librosa.frames_to_time(onset_frames, sr=self.sr)
            intervals = np.diff(frame_times) if len(frame_times) > 1 else np.array([0.1])
            if len(intervals) > 0 and np.mean(intervals) > 0:
                regularity = float(np.clip(1 - np.std(intervals) / np.mean(intervals), 0.0, 1.0))
            else:
                regularity = 0.0
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(self.audio, frame_length=self._frame_length, hop_length=self._hop_length)))
            return {"onset_regularity": regularity, "zero_crossing_rate": zcr, "onset_count": int(len(onset_frames))}
        except Exception:
            return {"onset_regularity": 0.0, "zero_crossing_rate": 0.0, "onset_count": 0}

    def hnr_estimate(self):
        """Approximate harmonic-to-noise ratio via H/P separation energy."""
        if not LIBROSA_AVAILABLE:
            return {"hnr": 0.0}
        try:
            h, p = librosa.effects.hpss(self.audio)
            # energy ratio harmonic / (percussive + eps)
            h_energy = np.mean(h ** 2)
            p_energy = np.mean(p ** 2) + 1e-12
            hnr = float(h_energy / p_energy)
            return {"hnr": hnr}
        except Exception:
            return {"hnr": 0.0}

    def speech_ratio(self):
        """Estimate fraction of frames that contain speech-like energy."""
        try:
            rms = librosa.feature.rms(y=self.audio, frame_length=self._frame_length, hop_length=self._hop_length)[0]
            threshold = np.percentile(rms, 30)  # conservative threshold
            speech_frames = np.sum(rms > threshold)
            ratio = float(speech_frames / max(1, len(rms)))
            return {"speech_ratio": ratio}
        except Exception:
            return {"speech_ratio": 0.0}

    def extract_all(self) -> Dict[str, float]:
        """Run all extractors and return a flat feature dict."""
        self.normalize()
        feats = {}
        feats.update(self.rms_stats())
        feats.update(self.spectral_stats())
        feats.update(self.mfcc_stats())
        feats.update(self.pitch_stats())
        feats.update(self.temporal_stats())
        feats.update(self.hnr_estimate())
        feats.update(self.speech_ratio())
        # Add audio length (seconds)
        try:
            feats["duration"] = float(len(self.audio) / max(1, self.sr))
        except Exception:
            feats["duration"] = 0.0
        return feats


# -------------------- Classifier: ensemble & calibration --------------------


class VoiceClassifier:
    """
    Ensemble-like rule-based classifier with richer feature contributions and calibration.
    Produces:
      - classification: "AI_GENERATED" or "HUMAN"
      - confidence: probability-like [0,1]
      - reliability: [0,1] indicating how trustworthy this prediction is (long audio, many features)
      - contributions: per-feature influence on the raw score
    """

    def __init__(self, features: Dict[str, float]):
        self.features = features or {}
        # Tunable weights (hand-tuned heuristics)
        self.weights = {
            "pitch_consistency": 3.5,
            "pitch_jitter": 4.0,
            "spectral_flux_mean": 2.5,
            "mfcc_var_mean": 2.5,
            "onset_regularity": 2.0,
            "energy_variation": 1.5,  # derived from rms_std
            "zero_crossing_rate": 1.0,
            "hnr": 1.5,
            "speech_ratio": 1.0,
            "duration": 0.5,
            "spectral_rolloff_mean": 0.8,
            "spectral_centroid_mean": 0.6,
        }
        # Feature directions: +1 means higher -> more AI, -1 means higher -> more Human
        self.directions = {
            "pitch_consistency": +1.0,  # very stable pitch often AI
            "pitch_jitter": +1.0,       # very high or very low jitter => AI artifacts
            "spectral_flux_mean": +1.0,
            "mfcc_var_mean": +1.0,      # low variance => repetitive AI -> treat by invert logic later
            "onset_regularity": +1.0,
            "energy_variation": -1.0,   # high variation => human (so negative direction for AI)
            "zero_crossing_rate": +1.0,
            "hnr": -1.0,                # high HNR suggests clean harmonic voice (often human) -> negative for AI
            "speech_ratio": -1.0,       # long continuous speech could be AI (but ambiguous) -> small weight
            "duration": -1.0,           # longer -> more reliable -> reduces AI extremeness (used in reliability)
            "spectral_rolloff_mean": +1.0,
            "spectral_centroid_mean": +1.0,
        }
        # Calibration factor for sigmoid: higher -> steeper transition
        self.sigmoid_scale = 2.5
        # Add thresholding behavior for some metrics (human ranges)
        self.human_expected = {
            # approximate human ranges; used to convert raw value to normalized deviation
            "pitch_jitter": (0.005, 0.035),
            "mfcc_var_mean": (0.35, 0.75),
            "pitch_consistency": (0.3, 0.85),
            "speech_ratio": (0.2, 0.95),
            "hnr": (0.5, 10.0),
        }

    def _norm_dev(self, val: float, human_range: Tuple[float, float]) -> float:
        """Compute normalized deviation from human range: 0 inside range, grows outside."""
        lo, hi = human_range
        if lo <= val <= hi:
            return 0.0
        # proportion outside
        if val < lo:
            return float((lo - val) / max(1e-6, abs(lo)))  # relative shortfall
        return float((val - hi) / max(1e-6, abs(hi)))

    def compute_contributions(self) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Compute raw AI score (sum of weighted feature signals), return:
         - raw_score
         - contributions per feature
         - normalized_signals per feature (for debugging)
        """
        raw_score = 0.0
        contributions = {}
        signals = {}

        # Derived features
        energy_variation = float(self.features.get("rms_std", 0.0))
        mfcc_var_mean = float(self.features.get("mfcc_var_mean", 0.0))
        pitch_consistency = float(self.features.get("pitch_consistency", 0.0))
        pitch_jitter = float(self.features.get("pitch_jitter", 0.0))
        spectral_flux = float(self.features.get("spectral_flux_mean", 0.0))
        onset_reg = float(self.features.get("onset_regularity", 0.0))
        zcr = float(self.features.get("zero_crossing_rate", 0.0))
        hnr = float(self.features.get("hnr", 0.0))
        speech_ratio = float(self.features.get("speech_ratio", 0.0))
        duration = float(self.features.get("duration", 0.0))
        rolloff = float(self.features.get("spectral_rolloff_mean", 0.0))
        centroid = float(self.features.get("spectral_centroid_mean", 0.0))

        # Signals: map raw measures into a [-1, +1] where +1 favors AI, -1 favors Human
        # 1) Pitch consistency: too high (near 1.0) => AI; very low => human
        sig_pitch_cons = np.clip((pitch_consistency - 0.6) * 2.0, -1.0, 1.0)
        signals["pitch_consistency"] = float(sig_pitch_cons)

        # 2) Pitch jitter: both extremes may indicate AI; convert to distance from human median
        pj_med = 0.02
        sig_pitch_jitter = np.clip((pitch_jitter - pj_med) / max(1e-6, pj_med) , -1.0, 2.0)
        signals["pitch_jitter"] = float(sig_pitch_jitter)

        # 3) Spectral flux: high flux indicates artifacts/noise (AI)
        sig_flux = np.clip((spectral_flux - 0.15) / 0.15, -1.0, 2.0)
        signals["spectral_flux_mean"] = float(sig_flux)

        # 4) MFCC variance: low variance -> more AI; invert so low -> +1
        sig_mfcc = np.clip((0.5 - mfcc_var_mean) / 0.5, -1.0, 2.0)
        signals["mfcc_var_mean"] = float(sig_mfcc)

        # 5) Onset regularity: high regularity (near 1) -> AI
        sig_onset = np.clip((onset_reg - 0.6) * 2.0, -1.0, 1.5)
        signals["onset_regularity"] = float(sig_onset)

        # 6) Energy variation: high std -> human -> negative for AI
        sig_energy_var = np.clip((energy_variation - 0.03) / 0.05, -2.0, 1.0)
        signals["energy_variation"] = float(sig_energy_var)

        # 7) Zero crossing rate (ZCR): high -> noise/artifacts -> AI
        sig_zcr = np.clip((zcr - 0.10) / 0.05, -1.0, 2.0)
        signals["zero_crossing_rate"] = float(sig_zcr)

        # 8) HNR: lower HNR -> noisy/artificial -> AI (so invert)
        sig_hnr = np.clip((1.0 - (hnr / (hnr + 1.0))) * 2.0, -1.0, 2.0)
        signals["hnr"] = float(sig_hnr)

        # 9) Speech ratio: very low speech ratio (fragmented) can be problematic; mild signal
        sig_speech_ratio = np.clip((0.6 - speech_ratio) * 1.5, -1.0, 1.0)
        signals["speech_ratio"] = float(sig_speech_ratio)

        # 10) spectral rolloff/centroid: high values may indicate AI artifacts depending on voice
        sig_rolloff = np.clip((rolloff - 3000) / 3000, -1.0, 2.0)
        signals["spectral_rolloff_mean"] = float(sig_rolloff)
        sig_centroid = np.clip((centroid - 2000) / 2000, -1.0, 2.0)
        signals["spectral_centroid_mean"] = float(sig_centroid)

        # Weighted sum
        for fname, weight in self.weights.items():
            sig = signals.get(fname, 0.0)
            direction = self.directions.get(fname, +1.0)
            contrib = float(weight * direction * sig)
            contributions[fname] = contrib
            raw_score += contrib

        # Bonus: penalize if duration too short (low reliability) by shrinking raw_score magnitude
        duration_sec = duration
        if duration_sec < 1.0:
            raw_score *= 0.2
        elif duration_sec < 3.0:
            raw_score *= 0.6

        return float(raw_score), contributions, signals

    def score_and_confidence(self) -> Dict[str, Any]:
        raw_score, contributions, signals = self.compute_contributions()
        # Map raw_score to [0,1] via scaled sigmoid
        try:
            p = 1.0 / (1.0 + math.exp(-raw_score / max(1e-6, self.sigmoid_scale)))
        except OverflowError:
            p = 1.0 if raw_score > 0 else 0.0
        confidence = float(np.clip(p, 0.0, 1.0))

        # Reliability: estimate based on duration, feature coverage, and backend availability
        dur = float(self.features.get("duration", 0.0))
        duration_factor = np.clip((dur / 10.0), 0.0, 1.0)  # 10s or more -> full
        feature_presence = len([v for v in self.features.values() if v is not None])
        feature_factor = np.clip(feature_presence / 12.0, 0.2, 1.0)
        # check if essential features are present (mfcc, pitch)
        essential_ok = (self.features.get("mfcc_var_mean", 0) > 0) and (self.features.get("pitch_consistency", 0) >= 0)
        essential_factor = 1.0 if essential_ok else 0.6
        reliability = float(np.clip(0.4 * duration_factor + 0.4 * feature_factor + 0.2 * essential_factor, 0.0, 1.0))

        # Adjust final confidence by reliability: keep confidence but also report calibrated_confidence
        # calibrated_confidence blends the raw confidence with 0.5 based on uncertainty
        calibrated_confidence = float(reliability * confidence + (1.0 - reliability) * 0.5)

        classification = "AI_GENERATED" if calibrated_confidence >= 0.50 else "HUMAN"

        # Build explanation with top contributors
        sorted_contrib = sorted(contributions.items(), key=lambda x: -abs(x[1]))
        top = sorted_contrib[:4]
        explanation_parts = []
        for feat, c in top:
            direction = "AI-like" if c > 0 else "Human-like"
            explanation_parts.append(f"{feat}={c:.2f} ({direction})")
        explanation = "; ".join(explanation_parts) if explanation_parts else "No strong indicators"

        return {
            "raw_score": raw_score,
            "confidence": float(confidence),
            "calibrated_confidence": float(calibrated_confidence),
            "reliability": float(reliability),
            "classification": classification,
            "contributions": contributions,
            "signals": signals,
            "explanation": explanation,
        }


# -------------------- Flask routes & main handler ---------------------------


@app.before_request
def log_request():
    try:
        logger.info("Incoming %s %s (len=%s)", request.method, request.path, request.headers.get("Content-Length"))
    except Exception:
        pass


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "message": "Voice Detection API running",
        "librosa_available": LIBROSA_AVAILABLE,
        "soundfile_available": SOUND_FILE_AVAILABLE,
        "endpoints": {"detection": "/api/voice-detection", "health": "/api/health", "stats": "/api/stats"},
    }), 200


@app.route("/api/voice-detection", methods=["GET", "POST", "OPTIONS", "HEAD"])
@app.route("/api/voice-detection/", methods=["GET", "POST", "OPTIONS", "HEAD"])
def detect_voice():
    if request.method == "OPTIONS":
        return ("", 204)
    if request.method in ("GET", "HEAD"):
        return jsonify({
            "status": "ok",
            "message": "Send POST with JSON {language, audioFormat, audioBase64} or multipart file 'file' and header x-api-key"
        }), 200

    try:
        api_key = extract_api_key_from_headers()
        if not check_api_key(api_key):
            return jsonify({"status": "error", "message": "Invalid API key"}), 401

        content_type = request.content_type or ""
        language = None
        audio_format = None
        audio_bytes = None

        if "application/json" in content_type or request.is_json:
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"status": "error", "message": "Invalid or empty JSON body"}), 400
            language = (data.get("language") or data.get("Language") or "").strip().title()
            audio_format = (data.get("audioFormat") or data.get("format") or "").lower().lstrip(".")
            audio_field = data.get("audioBase64") or data.get("audio_base64") or data.get("audio")
            if not audio_field:
                return jsonify({"status": "error", "message": "Missing audioBase64 field in JSON"}), 400
            audio_bytes, err = tolerant_base64_to_bytes(audio_field)
            if err:
                return jsonify({"status": "error", "message": err}), 400

        elif "multipart/form-data" in content_type:
            f = request.files.get("file") or request.files.get("audioFile") or request.files.get("audio")
            if not f:
                return jsonify({"status": "error", "message": "Missing file in multipart/form-data"}), 400
            try:
                audio_bytes = f.read()
            except Exception:
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False)
                    f.save(tmp.name)
                    tmp.close()
                    with open(tmp.name, "rb") as rf:
                        audio_bytes = rf.read()
                except Exception:
                    return jsonify({"status": "error", "message": "Failed to read uploaded file"}), 400
                finally:
                    try:
                        os.unlink(tmp.name)
                    except Exception:
                        pass
            language = (request.form.get("language") or request.form.get("lang") or "").strip().title()
            audio_format = (request.form.get("format") or request.form.get("audioFormat") or "").lower().lstrip(".")
        else:
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

        if not language:
            return jsonify({"status": "error", "message": "Missing language"}), 400
        if language not in LANGUAGES:
            return jsonify({"status": "error", "message": f"Unsupported language: {language}"}), 400

        if not audio_format:
            audio_format = "mp3"
        if audio_format.lower().lstrip(".") != "mp3":
            return jsonify({"status": "error", "message": "Only MP3 supported (tester expects .mp3)"}), 400

        if not audio_bytes:
            return jsonify({"status": "error", "message": "No audio data available"}), 400

        audio, sr, load_err = load_audio_from_bytes(audio_bytes)
        try:
            del audio_bytes
        except Exception:
            pass
        gc.collect()
        if load_err:
            logger.warning("Audio load failed: %s", load_err)
            return jsonify({"status": "error", "message": load_err}), 400

        if audio is None or sr is None or len(audio) < int(sr * 0.25):
            return jsonify({"status": "error", "message": "Audio too short"}), 400

        # Extract features
        extractor = FeatureExtractor(audio, sr)
        feats = extractor.extract_all()
        # Clean up heavy objects early
        try:
            del audio, extractor
        except Exception:
            pass
        gc.collect()

        # Classify
        classifier = VoiceClassifier(feats)
        scoring = classifier.score_and_confidence()

        # Compose response with details: classification, calibrated confidence, reliability,
        # and per-feature contributions/signals for transparency to the tester.
        response = {
            "status": "success",
            "language": language,
            "classification": scoring["classification"],
            "confidenceScore": round(float(scoring["calibrated_confidence"]), 3),
            "rawConfidence": round(float(scoring["confidence"]), 3),
            "reliability": round(float(scoring["reliability"]), 3),
            "explanation": scoring["explanation"],
            "features": feats,
            "contributions": scoring["contributions"],
            "signals": scoring["signals"],
            "model": "heuristic-ensemble-v2",
            "timestamp": datetime.utcnow().isoformat(),
        }

        # log summary
        logger.info("Classified: %s conf=%.3f reliability=%.3f lang=%s", scoring["classification"],
                    scoring["calibrated_confidence"], scoring["reliability"], language)

        # Save light-weight request history
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
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "supported_languages": sorted(list(LANGUAGES)),
        "librosa": LIBROSA_AVAILABLE,
        "soundfile": SOUND_FILE_AVAILABLE,
    }), 200


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
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Voice Detection API on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)