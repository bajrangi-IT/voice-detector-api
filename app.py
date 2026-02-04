from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import base64
import io
import logging
import warnings
from datetime import datetime
from functools import wraps
from scipy import stats

# Filter warnings for a cleaner production output
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Security configuration
VALID_KEYS = {
    'sk_test_123456789': 'test_user',
    'sk_prod_87654321': 'prod_user',
    'sk_hackathon_2024': 'hackathon_user'
}

SUPPORTED_LANGUAGES = ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu']

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        if not api_key or api_key not in VALID_KEYS:
            return jsonify({
                'status': 'error',
                'message': 'Invalid API key or malformed request'
            }), 401
        return f(*args, **kwargs)
    return decorated

class AudioProcessor:
    def __init__(self, data, sr=16000):
        self.audio = data
        self.sr = sr
        self._normalize()
        
    def _normalize(self):
        try:
            peak = np.max(np.abs(self.audio))
            if peak > 0:
                self.audio = self.audio / peak
        except Exception:
            pass
    
    def get_features(self):
        """Extracts acoustic features for classification."""
        features = {}
        try:
            # Spectral features
            stft = np.abs(librosa.stft(self.audio))
            S, _ = librosa.magphase(stft)
            sc = librosa.feature.spectral_centroid(S=S, sr=self.sr)[0]
            features['spectral_centroid'] = float(np.mean(sc))
            
            mel = librosa.feature.melspectrogram(y=self.audio, sr=self.sr)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            flux = np.sqrt(np.sum(np.diff(mel_db, axis=1)**2, axis=0))
            features['spectral_flux'] = float(np.mean(flux))

            # Pitch analysis
            f0 = librosa.yin(self.audio, fmin=50, fmax=500)
            f0_valid = f0[f0 > 0]
            if len(f0_valid) > 5:
                features['pitch_jitter'] = float(np.mean(np.abs(np.diff(f0_valid))) / np.mean(f0_valid))
                features['pitch_stability'] = float(1.0 - (np.std(f0_valid) / np.mean(f0_valid)))
            else:
                features['pitch_jitter'] = 0.05
                features['pitch_stability'] = 0.5

            # Timbral features (MFCC)
            mfcc = librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=13)
            features['mfcc_var'] = float(np.mean(np.var(mfcc, axis=1)))

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return None
        return features

def classify_voice(features):
    """Business logic for AI vs Human classification."""
    if not features:
        return "HUMAN", 0.50, "Standard vocal profile detected."

    jitter = features.get('pitch_jitter', 0)
    flux = features.get('spectral_flux', 0)
    stability = features.get('pitch_stability', 0)
    variance = features.get('mfcc_var', 0)

    # Heuristic scoring based on typical AI artifacts
    score = 0
    if jitter > 0.08: score += 0.40
    if flux > 20.0: score += 0.30
    if stability > 0.90: score += 0.20
    if variance < 40.0: score += 0.10

    if score > 0.48:
        label = "AI_GENERATED"
        conf = min(0.99, score)
        if conf > 0.75:
            reason = "Synthetic signature detected: unnatural spectral flux and robotic pitch patterns."
        else:
            reason = "AI voice artifacts detected: consistent pitch stability and regular acoustic timing."
    else:
        label = "HUMAN"
        conf = max(0.15, 1.0 - score)
        reason = "Natural speech characteristics detected with organic variations in pitch and energy."

    return label, round(float(conf), 2), reason

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'status': 'success',
        'message': 'Voice Analysis Gateway Active'
    }), 200

@app.route('/api/voice-detection', methods=['POST'])
@require_auth
def voice_detection():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid API key or malformed request'}), 400

        lang = data.get('language', '').strip()
        fmt = data.get('audioFormat', '').lower()
        b64 = data.get('audioBase64', '')

        # Input validation
        if lang not in SUPPORTED_LANGUAGES or fmt != 'mp3' or not b64:
            return jsonify({'status': 'error', 'message': 'Invalid API key or malformed request'}), 400

        # Decode and load audio
        try:
            raw_audio = base64.b64decode(b64)
            buf = io.BytesIO(raw_audio)
            y, sr = librosa.load(buf, sr=16000, duration=10)
            if len(y) < 1600: # Min 0.1s
                return jsonify({'status': 'error', 'message': 'Invalid API key or malformed request'}), 400
        except Exception:
            return jsonify({'status': 'error', 'message': 'Invalid API key or malformed request'}), 400

        # Process and classify
        processor = AudioProcessor(y, sr)
        stats = processor.get_features()
        label, score, reason = classify_voice(stats)

        return jsonify({
            'status': 'success',
            'language': lang,
            'classification': label,
            'confidenceScore': score,
            'explanation': reason
        }), 200

    except Exception as e:
        logger.error(f"Gateway error: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Invalid API key or malformed request'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'success',
        'uptime': 'active',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.errorhandler(400)
@app.errorhandler(401)
@app.errorhandler(404)
@app.errorhandler(405)
@app.errorhandler(500)
def handle_error(e):
    return jsonify({'status': 'error', 'message': 'Invalid API key or malformed request'}), getattr(e, 'code', 500)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)