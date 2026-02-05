#!/usr/bin/env python3
"""
Optimized Voice Detection API (post-v9 improvements)

Key optimizations:
- MP3 -> WAV transcode with ffmpeg to 16kHz mono, trim to 8s (fast, consistent input).
- sr=16000, smaller frame/hop sizes for faster features.
- LRU cache for repeated audio submissions (sha256).
- Fast-path lightweight scoring + full-path richer scoring.
- Timing logs for each stage.
- Robust error handling and deterministic JSON responses.

Requires ffmpeg installed in runtime (use Dockerfile below).
"""

import os
import io
import time
import json
import gc
import base64
import hashlib
import logging
import shutil
import subprocess
from collections import OrderedDict
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS

# Optional backends
try:
    import soundfile as sf
    SOUND_FILE_AVAILABLE = True
except Exception:
    sf = None
    SOUND_FILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    librosa = None
    LIBROSA_AVAILABLE = False

import numpy as np
import binascii
import re

# --- Config ---
MAX_SECONDS = 8             # trim/limit audio length (seconds)
TARGET_SR = 16000           # sample rate for processing
CHANNELS = 1
FFMPEG_TIMEOUT = 12
CACHE_SIZE = 128            # number of recent audio hashes to cache
FAST_ONLY = False           # if True, return fast predictions only (for extreme low-latency)

API_KEYS = {
    "sk_test_123456789": "test_user",
    "sk_prod_87654321": "prod_user",
}
ENV_KEY = os.environ.get("API_KEY")
if ENV_KEY:
    API_KEYS.setdefault(ENV_KEY, "env_user")

LANGUAGES = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vd-optimized")

# --- App ---
app = Flask(__name__)
app.url_map.strict_slashes = False
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60MB
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)

# --- Simple in-memory LRU cache for audio hash -> result ---
class SimpleLRU:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.data = OrderedDict()
    def get(self, key):
        try:
            val = self.data.pop(key)
            self.data[key] = val
            return val
        except KeyError:
            return None
    def set(self, key, value):
        if key in self.data:
            self.data.pop(key)
        elif len(self.data) >= self.capacity:
            self.data.popitem(last=False)
        self.data[key] = value

cache = SimpleLRU(CACHE_SIZE)

# --- Helpers ---
def extract_api_key_from_headers() -> Optional[str]:
    h = request.headers.get("x-api-key") or request.headers.get("X-API-KEY")
    if h:
        return h.strip()
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None

def check_api_key(key: Optional[str]) -> bool:
    return bool(key and key in API_KEYS)

def tolerant_base64_to_bytes(s: object) -> Tuple[Optional[bytes], Optional[str]]:
    if s is None:
        return None, "Audio payload missing"
    if isinstance(s, (bytes, bytearray)):
        try:
            s = s.decode("utf-8", errors="ignore")
        except Exception:
            s = str(s)
    if not isinstance(s, str):
        return None, "Audio payload must be text base64"
    s = s.strip()
    # strip data: URI if present
    m = re.match(r"^\s*data:audio\/[a-z0-9.+-]+;base64,", s, flags=re.I)
    if m:
        s = s[m.end():]
    s = "".join(s.split())
    cleaned = re.sub(r"[^A-Za-z0-9+/=_\-]", "", s)
    cleaned = cleaned.replace("-", "+").replace("_", "/")
    pad_len = (-len(cleaned)) % 4
    if pad_len:
        cleaned += "=" * pad_len
    if len(cleaned) < 16:
        return None, "Audio base64 too short"
    try:
        b = base64.b64decode(cleaned, validate=False)
        if not b:
            return None, "Decoded audio empty"
        return b, None
    except Exception as e:
        return None, f"Base64 decode failed: {e}"

def ffmpeg_to_wav_bytes(in_bytes: bytes, sr=TARGET_SR, channels=CHANNELS, max_seconds=MAX_SECONDS, timeout=FFMPEG_TIMEOUT) -> Tuple[Optional[bytes], Optional[str], Optional[float]]:
    """Transcode to WAV with ffmpeg, downsample & trim. Returns (wav_bytes, error, elapsed_s)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None, "ffmpeg not installed", None
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
            "-ar", str(sr), "-ac", str(channels), "-t", str(max_seconds), "-f", "wav", "pipe:1"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(args, input=in_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        t1 = time.perf_counter()
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode("utf-8", errors="ignore")
            return None, f"ffmpeg failed: {err.strip()[:200]}", t1 - t0
        return proc.stdout, None, t1 - t0
    except subprocess.TimeoutExpired:
        return None, "ffmpeg timed out", None
    except Exception as e:
        return None, f"ffmpeg error: {e}", None

def load_audio_from_wav_bytes(wav_bytes: bytes, sr=TARGET_SR) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """Read WAV bytes into numpy mono float32 array. Prefer soundfile then librosa."""
    if not wav_bytes:
        return None, None, "No wav bytes"
    if SOUND_FILE_AVAILABLE:
        try:
            bio = io.BytesIO(wav_bytes)
            data, sr_read = sf.read(bio, dtype="float32")
            if data is None or len(data) == 0:
                return None, None, "soundfile read empty"
            if getattr(data, "ndim", 1) > 1:
                data = np.mean(data, axis=1)
            return data.astype(np.float32), int(sr_read), None
        except Exception as e:
            logger.debug("soundfile read failed: %s", str(e))
    if LIBROSA_AVAILABLE:
        try:
            bio = io.BytesIO(wav_bytes)
            data, sr_read = librosa.load(bio, sr=sr, mono=True, duration=MAX_SECONDS)
            if data is None or len(data) == 0:
                return None, None, "librosa load empty"
            return data.astype(np.float32), int(sr_read), None
        except Exception as e:
            logger.debug("librosa read failed: %s", str(e))
    return None, None, "No audio backend available to read WAV (install soundfile or librosa)"

# --- Lightweight fast features & scorer (cheap path) ---
def compute_fast_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute a small set of cheap features for a fast prediction."""
    try:
        rms = np.sqrt(np.mean(audio**2))
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) ) if len(audio) > 1 else 0.0
        energy_std = float(np.std(audio))
        duration = float(len(audio) / sr)
        # cheap spectral centroid using small FFT via numpy (approx)
        try:
            centroid = float(np.mean(np.abs(np.fft.rfft(audio))[:512]))
        except Exception:
            centroid = 0.0
        return {"rms": float(rms), "zcr": zcr, "energy_std": energy_std, "duration": duration, "centroid": centroid}
    except Exception:
        return {"rms": 0.0, "zcr": 0.0, "energy_std": 0.0, "duration": 0.0, "centroid": 0.0}

def fast_score(features: Dict[str, float]) -> Tuple[str, float]:
    """
    Very lightweight scoring: heuristic based on rms, zcr, energy_std.
    Returns (classification, confidence).
    """
    rms = features.get("rms", 0.0)
    zcr = features.get("zcr", 0.0)
    energy_std = features.get("energy_std", 0.0)
    # heuristic: very low rms + low energy std -> synthetic? (depends)
    score = 0.0
    # low dynamic range (energy_std small) increases AI-like score
    score += (0.5 - min(energy_std, 0.5)) * 2.0
    # extreme zcr may indicate artifacts
    score += max(0.0, (zcr - 0.1) * 2.0)
    # normalize
    p = 1.0 / (1.0 + np.exp(-score))
    classification = "AI_GENERATED" if p >= 0.5 else "HUMAN"
    return classification, float(np.clip(p, 0.0, 1.0))

# --- Full extractor (reduced-cost params) ---
def extract_features_fastpath(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute mid-cost features with reduced sizes for speed."""
    # parameters tuned for speed
    frame_length = 1024
    hop_length = 256
    n_mfcc = 13
    n_mels = 64
    feats = {}
    try:
        # MFCC variance (cheap)
        if LIBROSA_AVAILABLE:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
            feats["mfcc_var_mean"] = float(np.mean(np.var(mfcc, axis=1)))
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
            feats["spectral_centroid_mean"] = float(np.mean(centroid))
            # spectral flux (from mel)
            S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
            S_db = librosa.power_to_db(S, ref=np.max)
            flux = np.sqrt(np.sum(np.diff(S_db, axis=1)**2, axis=0))
            feats["spectral_flux_mean"] = float(np.mean(flux)) if len(flux)>0 else 0.0
            # pitch estimate (yin) on shorter frames
            try:
                f0 = librosa.yin(audio, fmin=50, fmax=500, frame_length=frame_length, hop_length=hop_length, trough_threshold=0.1)
                f0_valid = f0[f0>0]
                feats["pitch_consistency"] = float(np.clip(1.0 - np.std(f0_valid/np.mean(f0_valid)) if len(f0_valid)>2 else 0.0, 0.0, 1.0))
                feats["pitch_jitter"] = float(np.mean(np.abs(np.diff(f0_valid))) / (np.mean(f0_valid)+1e-9) if len(f0_valid)>2 else 0.0)
            except Exception:
                feats["pitch_consistency"] = 0.0
                feats["pitch_jitter"] = 0.0
        else:
            # fallback cheap approximations
            feats["mfcc_var_mean"] = 0.0
            feats["spectral_centroid_mean"] = 0.0
            feats["spectral_flux_mean"] = 0.0
            feats["pitch_consistency"] = 0.0
            feats["pitch_jitter"] = 0.0
        # add rms and zcr
        feats.update(compute_fast_features(audio, sr))
    except Exception as e:
        logger.debug("extract_features_fastpath failed: %s", str(e))
        # fallback zeroes
        feats.update({"mfcc_var_mean":0.0, "spectral_flux_mean":0.0, "pitch_consistency":0.0, "pitch_jitter":0.0})
    return feats

# --- Classifier: simple calibrated ensemble (fast) ---
def score_features_ensemble(feats: Dict[str, float]) -> Dict[str, Any]:
    # simple weighted linear scoring (hand-tuned), then sigmoid calibration
    # weights chosen to be modest and robust
    weights = {
        "pitch_consistency": 3.0,
        "pitch_jitter": 4.0,
        "spectral_flux_mean": 2.0,
        "mfcc_var_mean": -2.5,  # negative: higher mfcc var -> human
        "energy_std": -1.5,
        "zcr": 1.0,
    }
    raw = 0.0
    contributions = {}
    for k,w in weights.items():
        v = float(feats.get(k, 0.0))
        contrib = w * v
        contributions[k] = contrib
        raw += contrib
    # map to probability
    p = 1.0 / (1.0 + np.exp(-raw / 3.0))
    calibrated = float(np.clip(p, 0.0, 1.0))
    classification = "AI_GENERATED" if calibrated >= 0.5 else "HUMAN"
    return {"raw": raw, "probability": calibrated, "classification": classification, "contributions": contributions}

# --- Endpoint ---
@app.route("/api/voice-detection", methods=["POST","OPTIONS","GET","HEAD"])
def detect_voice():
    if request.method == "OPTIONS":
        return ("", 204)
    if request.method in ("GET","HEAD"):
        return jsonify({"status":"ok","message":"POST JSON {language,audioFormat,audioBase64} or multipart 'file' with x-api-key"}), 200

    t_start = time.perf_counter()
    # auth
    api_key = extract_api_key_from_headers()
    if not check_api_key(api_key):
        return jsonify({"status":"error","message":"Invalid API key"}), 401

    content_type = request.content_type or ""
    language = None
    audio_format = None
    audio_bytes = None

    # parse input
    if "application/json" in content_type or request.is_json:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"status":"error","message":"Invalid/empty JSON"}), 400
        language = (data.get("language") or data.get("Language") or "").strip().title()
        audio_format = (data.get("audioFormat") or data.get("format") or "").lower().lstrip(".")
        a_field = data.get("audioBase64") or data.get("audio_base64") or data.get("audio")
        if not a_field:
            return jsonify({"status":"error","message":"Missing audioBase64"}), 400
        audio_bytes, err = tolerant_base64_to_bytes(a_field)
        if err:
            return jsonify({"status":"error","message":err}), 400
    elif "multipart/form-data" in content_type:
        f = request.files.get("file") or request.files.get("audioFile") or request.files.get("audio")
        if not f:
            return jsonify({"status":"error","message":"Missing file in multipart"}), 400
        try:
            audio_bytes = f.read()
        except Exception:
            return jsonify({"status":"error","message":"Failed to read uploaded file"}), 400
        language = (request.form.get("language") or request.form.get("lang") or "").strip().title()
        audio_format = (request.form.get("format") or request.form.get("audioFormat") or "").lower().lstrip(".")
    else:
        data = request.get_json(silent=True)
        if data:
            language = (data.get("language") or data.get("Language") or "").strip().title()
            audio_format = (data.get("audioFormat") or data.get("format") or "").lower().lstrip(".")
            a_field = data.get("audioBase64") or data.get("audio_base64") or data.get("audio")
            if a_field:
                audio_bytes, err = tolerant_base64_to_bytes(a_field)
                if err:
                    return jsonify({"status":"error","message":err}), 400
        else:
            return jsonify({"status":"error","message":"Unsupported Content-Type"}), 415

    if not language:
        return jsonify({"status":"error","message":"Missing language"}), 400
    if language not in LANGUAGES:
        return jsonify({"status":"error","message":"Unsupported language"}), 400

    if not audio_format:
        audio_format = "mp3"
    if audio_format.lower().lstrip(".") != "mp3":
        return jsonify({"status":"error","message":"Only MP3 input supported"}), 400
    if not audio_bytes:
        return jsonify({"status":"error","message":"No audio data"}), 400

    # Cache key
    key_hash = hashlib.sha256(audio_bytes).hexdigest()
    cached = cache.get(key_hash)
    if cached:
        # return cached result immediately
        elapsed = time.perf_counter() - t_start
        resp = dict(cached)
        resp["_cached"] = True
        resp["_latency_s"] = round(elapsed, 3)
        return jsonify(resp), 200

    # Transcode via ffmpeg -> WAV 16k mono trimmed
    t0 = time.perf_counter()
    wav_bytes, ff_err, ff_time = ffmpeg_to_wav_bytes(audio_bytes, sr=TARGET_SR, channels=CHANNELS, max_seconds=MAX_SECONDS)
    t1 = time.perf_counter()
    if ff_err:
        logger.warning("ffmpeg error: %s", ff_err)
        return jsonify({"status":"error","message": f"Audio decoding failed: {ff_err}"}), 400

    # load wav bytes to numpy
    audio, sr, load_err = load_audio_from_wav_bytes(wav_bytes, sr=TARGET_SR)
    t2 = time.perf_counter()
    if load_err:
        logger.warning("load audio failed: %s", load_err)
        return jsonify({"status":"error","message": load_err}), 400

    # Make sure audio length >= small threshold
    duration = len(audio) / sr
    if duration < 0.25:
        return jsonify({"status":"error","message":"Audio too short"}), 400

    # Fast features + fast score
    t_fp = time.perf_counter()
    fast_feats = compute_fast_features(audio, sr)
    fast_class, fast_conf = fast_score(fast_feats)
    t_fp2 = time.perf_counter()

    # If configured to return only fast results for ultra-low latency
    if FAST_ONLY:
        resp = {
            "status":"success",
            "language": language,
            "classification": fast_class,
            "confidenceScore": round(float(fast_conf),3),
            "fast_path": True,
            "features": fast_feats,
        }
        cache.set(key_hash, resp)
        resp["_latency_s"] = round(time.perf_counter() - t_start, 3)
        return jsonify(resp), 200

    # Full (reduced-cost) features and ensemble scoring
    t_feat0 = time.perf_counter()
    feats = extract_features_fastpath(audio, sr)
    t_feat1 = time.perf_counter()
    score = score_features_ensemble(feats)
    t_score = time.perf_counter()

    # Compose result
    resp = {
        "status": "success",
        "language": language,
        "classification": score["classification"],
        "confidenceScore": round(float(score["probability"]), 3),
        "fast_classification": fast_class,
        "fast_confidence": round(float(fast_conf),3),
        "features": feats,
        "contributions": score.get("contributions", {}),
        "timings": {
            "total_s": round(time.perf_counter() - t_start, 3),
            "ffmpeg_s": round(ff_time if ff_time else (t1 - t0), 3),
            "load_s": round(t2 - t1, 3),
            "fast_feats_s": round(t_fp2 - t_fp, 3),
            "feature_extract_s": round(t_feat1 - t_feat0, 3),
            "scoring_s": round(t_score - t_feat1, 3),
        },
        "model": "optimized-ensemble-v9->post",
        "timestamp": datetime.utcnow().isoformat()
    }

    # cache the response (light copy)
    try:
        cache.set(key_hash, dict(resp))
    except Exception:
        pass

    # cleanup
    try:
        del audio, wav_bytes
        gc.collect()
    except Exception:
        pass

    return jsonify(resp), 200

# simple health and stats endpoints
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status":"online","librosa":LIBROSA_AVAILABLE,"soundfile":SOUND_FILE_AVAILABLE}), 200

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status":"healthy","timestamp":datetime.utcnow().isoformat()}), 200

# Run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting optimized vd API on port %s", port)
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)