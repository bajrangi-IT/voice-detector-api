#!/usr/bin/env python3
"""
Voice Detection API - Detects AI-generated vs Human voices
Supports: Tamil, English, Hindi, Malayalam, Telugu
Author: Voice Detection Team
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import base64
import io
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
API_KEYS = {
    'sk_test_123456789': 'test_user',
    'sk_prod_87654321': 'prod_user'
}

LANGUAGES = ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu']

# Track requests
request_history = []


def check_api_key(api_key):
    """Validate API key"""
    return api_key in API_KEYS


def validate_request(data):
    """Validate incoming request"""
    if not data:
        return False, "No data provided"
    
    language = data.get('language', '').strip()
    audio_format = data.get('audioFormat', '').lower()
    audio_base64 = data.get('audioBase64', '')
    
    if language not in LANGUAGES:
        return False, f"Language '{language}' not supported"
    
    if audio_format != 'mp3':
        return False, "Only MP3 format is supported"
    
    if not audio_base64:
        return False, "No audio data provided"
    
    return True, (language, audio_format, audio_base64)


def decode_audio(audio_base64):
    """Decode base64 audio to bytes"""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        if len(audio_bytes) == 0:
            return None, "Empty audio data"
        return audio_bytes, None
    except Exception as e:
        logger.error(f"Base64 decode error: {e}")
        return None, "Failed to decode audio"


def load_audio(audio_bytes):
    """Load audio file using librosa"""
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio, sr = librosa.load(audio_file, sr=22050, duration=30)
        
        if len(audio) < sr * 0.3:  # Less than 0.3 seconds
            return None, None, "Audio too short"
        
        return audio, sr, None
    except Exception as e:
        logger.error(f"Audio loading error: {e}")
        return None, None, "Failed to load audio"


class FeatureExtractor:
    """Extract acoustic features from audio"""
    
    def __init__(self, audio, sr):
        self.audio = audio
        self.sr = sr
        self.features = {}
    
    def normalize(self):
        """Normalize audio to -1 to 1 range"""
        max_val = np.max(np.abs(self.audio))
        if max_val > 1e-6:
            self.audio = self.audio / max_val
    
    def extract_pitch_features(self):
        """Extract pitch-based features"""
        try:
            f0 = librosa.yin(self.audio, fmin=50, fmax=500, trough_threshold=0.1)
            f0_valid = f0[f0 > 0]
            
            if len(f0_valid) < 5:
                return {
                    'pitch_consistency': 0.0,
                    'pitch_jitter': 0.0
                }
            
            # Pitch consistency - how stable is the pitch
            f0_norm = f0_valid / np.mean(f0_valid)
            pitch_consistency = 1.0 - np.std(f0_norm)
            pitch_consistency = np.clip(pitch_consistency, 0, 1)
            
            # Pitch jitter - micro variations
            pitch_jitter = np.mean(np.abs(np.diff(f0_valid))) / np.mean(f0_valid)
            
            return {
                'pitch_consistency': float(pitch_consistency),
                'pitch_jitter': float(pitch_jitter)
            }
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            return {'pitch_consistency': 0.0, 'pitch_jitter': 0.0}
    
    def extract_spectral_features(self):
        """Extract spectral features"""
        try:
            # Mel spectrogram
            S = librosa.feature.melspectrogram(y=self.audio, sr=self.sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr)[0]
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=self.audio, sr=self.sr)
            
            # Spectral flux - how much spectrum changes
            flux = np.sqrt(np.sum(np.diff(S_db, axis=1)**2, axis=0))
            
            return {
                'spectral_centroid': float(np.mean(centroid)),
                'spectral_flux': float(np.mean(flux)),
                'spectral_contrast': float(np.mean(contrast))
            }
        except Exception as e:
            logger.warning(f"Spectral extraction failed: {e}")
            return {
                'spectral_centroid': 0.0,
                'spectral_flux': 0.0,
                'spectral_contrast': 0.0
            }
    
    def extract_mfcc_features(self):
        """Extract MFCC features - voice identity"""
        try:
            mfcc = librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=13)
            
            # MFCC variance tells us if voice is repetitive (AI) or varied (Human)
            mfcc_variance = float(np.mean(np.var(mfcc, axis=1)))
            
            return {
                'mfcc_variance': mfcc_variance
            }
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            return {'mfcc_variance': 0.0}
    
    def extract_temporal_features(self):
        """Extract timing-based features"""
        try:
            # Onset detection - detect speech segments
            onset_env = librosa.onset.onset_strength(y=self.audio, sr=self.sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, units='frames')
            
            frame_times = librosa.frames_to_time(onset_frames, sr=self.sr)
            intervals = np.diff(frame_times) if len(frame_times) > 1 else np.array([0.1])
            
            # How regular is the timing
            if len(intervals) > 0 and np.mean(intervals) > 0:
                regularity = 1 - np.clip(np.std(intervals) / np.mean(intervals), 0, 1)
            else:
                regularity = 0.0
            
            # Zero crossing rate - high frequency content
            zcr = np.mean(librosa.feature.zero_crossing_rate(self.audio))
            
            return {
                'onset_regularity': float(regularity),
                'zero_crossing_rate': float(zcr)
            }
        except Exception as e:
            logger.warning(f"Temporal extraction failed: {e}")
            return {
                'onset_regularity': 0.0,
                'zero_crossing_rate': 0.0
            }
    
    def extract_energy_features(self):
        """Extract energy-based features"""
        try:
            stft = np.abs(librosa.stft(self.audio))
            power = np.abs(stft) ** 2
            energy = np.sum(power, axis=0)
            
            # Energy ratio tells if volume is consistent (AI) or varies (Human)
            energy_ratio = np.std(energy) / (np.mean(energy) + 1e-6)
            
            return {
                'energy_ratio': float(energy_ratio)
            }
        except Exception as e:
            logger.warning(f"Energy extraction failed: {e}")
            return {'energy_ratio': 0.0}
    
    def extract_all(self):
        """Extract all features"""
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
        """Calculate how likely the voice is AI-generated"""
        score = 0.0
        
        # AI voices have unnaturally consistent pitch
        pitch_cons = self.features.get('pitch_consistency', 0.5)
        if pitch_cons > 0.90:
            score += 5.0
        elif pitch_cons > 0.85:
            score += 3.5
        elif pitch_cons < 0.75:
            score -= 3.0
        
        # AI voices have repetitive patterns (low MFCC variance)
        mfcc_var = self.features.get('mfcc_variance', 0.5)
        if mfcc_var < 0.35:
            score += 4.5
        elif mfcc_var > 0.55:
            score -= 3.0
        
        # AI voices have too regular timing
        onset_reg = self.features.get('onset_regularity', 0.5)
        if onset_reg > 0.85:
            score += 4.5
        elif onset_reg < 0.70:
            score -= 3.0
        
        # AI voices have smooth spectrum
        flux = self.features.get('spectral_flux', 0.2)
        if flux < 0.13:
            score += 4.0
        elif flux > 0.20:
            score -= 2.5
        
        # AI voices lack natural jitter
        jitter = self.features.get('pitch_jitter', 0.03)
        if jitter < 0.008:
            score += 3.5
        elif jitter > 0.045:
            score -= 2.5
        
        # AI voices have uniform energy
        energy = self.features.get('energy_ratio', 0.4)
        if energy < 0.20:
            score += 3.5
        elif energy > 0.55:
            score -= 2.5
        
        # AI voices lack high-frequency content
        zcr = self.features.get('zero_crossing_rate', 0.1)
        if zcr < 0.04:
            score += 2.5
        elif zcr > 0.12:
            score -= 2.0
        
        # AI voices have lower spectral contrast
        contrast = self.features.get('spectral_contrast', 0)
        if contrast < 5.0:
            score += 2.0
        elif contrast > 7.0:
            score -= 1.5
        
        return score
    
    def classify(self):
        """Classify the voice"""
        ai_score = self.score_ai_likelihood()
        
        # Convert to confidence (0-1 range)
        # Higher AI score means more likely AI
        confidence = 1.0 / (1.0 + np.exp(-ai_score / 5.0))  # Sigmoid function
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # Decide classification
        classification = 'AI_GENERATED' if confidence > 0.48 else 'HUMAN'
        
        # Generate explanation
        explanation = self.get_explanation(confidence)
        
        return {
            'classification': classification,
            'confidence': confidence,
            'explanation': explanation
        }
    
    def get_explanation(self, confidence):
        """Generate explanation for classification"""
        if confidence > 0.65:
            return "Strong AI voice characteristics: artificial pitch patterns and spectral smoothness detected"
        elif confidence > 0.50:
            return "AI voice characteristics detected: unnaturally consistent pitch and regular timing patterns"
        elif confidence > 0.48:
            return "Likely AI-generated voice with artificial voice characteristics"
        elif confidence < 0.35:
            return "Strong human speech characteristics with natural variation and complexity"
        else:
            return "Natural human voice detected with typical speech dynamics and variation"


# API Routes

@app.route('/', methods=['GET'])
def index():
    """Root route for verification"""
    return jsonify({
        'status': 'online',
        'message': 'Voice Detection API is running',
        'endpoints': {
            'detection': '/api/voice-detection',
            'health': '/api/health',
            'stats': '/api/stats'
        }
    }), 200


@app.route('/api/voice-detection', methods=['POST'])
def detect_voice():
    """Main endpoint for voice detection"""
    
    # Get API key from header
    api_key = request.headers.get('x-api-key')
    
    if not api_key or not check_api_key(api_key):
        logger.warning(f"Invalid API key attempt: {api_key}")
        return jsonify({
            'status': 'error',
            'message': 'Invalid API key or malformed request'
        }), 401
    
    # Get JSON data
    data = request.get_json()
    
    # Validate request
    is_valid, result = validate_request(data)
    if not is_valid:
        logger.warning(f"Invalid request: {result}")
        return jsonify({
            'status': 'error',
            'message': 'Invalid API key or malformed request'
        }), 400
    
    language, audio_format, audio_base64 = result
    
    # Decode audio
    audio_bytes, error = decode_audio(audio_base64)
    if error:
        logger.warning(f"Decode error: {error}")
        return jsonify({
            'status': 'error',
            'message': 'Invalid API key or malformed request'
        }), 400
    
    # Load audio
    audio, sr, error = load_audio(audio_bytes)
    if error:
        logger.warning(f"Load error: {error}")
        return jsonify({
            'status': 'error',
            'message': 'Invalid API key or malformed request'
        }), 400
    
    # Extract features
    try:
        extractor = FeatureExtractor(audio, sr)
        features = extractor.extract_all()
        logger.info(f"Features extracted for {language}")
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Invalid API key or malformed request'
        }), 500
    
    # Classify
    try:
        classifier = VoiceClassifier(features)
        result = classifier.classify()
        logger.info(f"Classification: {result['classification']} (confidence: {result['confidence']:.2f})")
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Invalid API key or malformed request'
        }), 500
    
    # Build response
    response = {
        'status': 'success',
        'language': language,
        'classification': result['classification'],
        'confidenceScore': round(result['confidence'], 2),
        'explanation': result['explanation']
    }
    
    # Log request
    request_history.append({
        'timestamp': datetime.now().isoformat(),
        'language': language,
        'classification': result['classification']
    })
    
    return jsonify(response), 200


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'supported_languages': LANGUAGES
    }), 200


@app.route('/api/stats', methods=['GET'])
def stats():
    """Statistics endpoint"""
    api_key = request.headers.get('x-api-key')
    
    if not api_key or not check_api_key(api_key):
        return jsonify({
            'status': 'error',
            'message': 'Invalid API key or malformed request'
        }), 401
    
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
    return jsonify({'status': 'error', 'message': 'Bad Request: Malformed JSON or missing data'}), 400


@app.errorhandler(401)
def unauthorized(error):
    return jsonify({'status': 'error', 'message': 'Unauthorized: Invalid API key'}), 401


@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Resource not found. Use /api/voice-detection or /api/health'}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'status': 'error', 'message': 'Method not allowed'}), 405


@app.errorhandler(500)
def server_error(error):
    logger.error(f"Internal Server Error: {error}")
    return jsonify({'status': 'error', 'message': 'Internal Server Error. Please check logs.'}), 500


# Main
if __name__ == '__main__':
    print("\n" + "="*80)
    print("Voice Detection API - Starting...")
    print("="*80)
    print("Supported Languages: Tamil, English, Hindi, Malayalam, Telugu")
    print("Endpoint: http://0.0.0.0:5000/api/voice-detection")
    print("="*80 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)