"""
Dog Breed Classification Web Application
==========================================
A beautiful Flask-based web app for dog breed prediction with behavioral insights.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import json
import numpy as np
from scipy.io import wavfile
import io
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import get_transforms
from src.model import get_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Breed names will be loaded dynamically from the fused dataset
BREED_NAMES = []

# Global model and behavioral data
model = None
transform = None
behavioral_data = None
device = None

# Audio model globals
audio_model = None
audio_config = None
audio_mel_fb = None
audio_window = None
AUDIO_CLASS_NAMES = ["cats", "dogs"]
AUDIO_LABELS = {"cats": "🐱 Cat", "dogs": "🐕 Dog"}


def load_model():
    """Load the trained model."""
    global model, transform, device, BREED_NAMES
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load breed names from fused dataset (which was used for training)
    fused_train_path = "data/fused/train"
    if os.path.exists(fused_train_path):
        BREED_NAMES = sorted(os.listdir(fused_train_path))
        print(f"Loaded {len(BREED_NAMES)} breed classes from fused dataset")
    else:
        # Fallback to regular train folder
        train_path = "data/train"
        if os.path.exists(train_path):
            BREED_NAMES = sorted(os.listdir(train_path))
            print(f"Loaded {len(BREED_NAMES)} breed classes from train dataset")
    
    num_classes = len(BREED_NAMES)
    print(f"Number of classes: {num_classes}")
    
    # Try to load the best model
    model_paths = [
        "models/mobilenet_v2_20260129_230009/best_model.pth",
        "models/resnet50_20260129_220924/best_model.pth",
    ]
    
    model_path = None
    model_name = None
    
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            if "mobilenet" in path:
                model_name = "mobilenet_v2"
            elif "resnet" in path:
                model_name = "resnet50"
            break
    
    if model_path is None:
        print("Warning: No trained model found! Using untrained model for demo.")
        model_name = "mobilenet_v2"
        model = get_model(model_name, num_classes=num_classes, pretrained=True)
    else:
        print(f"Loading model from: {model_path}")
        model = get_model(model_name, num_classes=num_classes, pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(device)
    model.eval()
    
    transform = get_transforms(image_size=224, is_training=False)
    print("Model loaded successfully!")


def load_behavioral_data():
    """Load behavioral data for breed insights."""
    global behavioral_data
    
    try:
        # Load the comprehensive dog data
        data_path = "data/behaviour/dog_data.xlsx"
        if os.path.exists(data_path):
            behavioral_data = pd.read_excel(data_path)
            print(f"Loaded behavioral data for {len(behavioral_data)} breeds")
        else:
            # Try CSV files
            csv_path = "data/behaviour/dog_breeds.csv"
            if os.path.exists(csv_path):
                behavioral_data = pd.read_csv(csv_path)
                print(f"Loaded behavioral data from CSV for {len(behavioral_data)} breeds")
            else:
                print("Warning: No behavioral data found")
                behavioral_data = None
    except Exception as e:
        print(f"Error loading behavioral data: {e}")
        behavioral_data = None


def normalize_breed_name(name: str) -> str:
    """Normalize breed name for matching."""
    # Remove common suffixes and prefixes
    name = name.lower().replace("_", " ").replace("-", " ")
    # Remove parenthetical info like "(Standard)", "(Miniature)"
    if "(" in name:
        name = name.split("(")[0].strip()
    # Remove common words that don't help matching
    remove_words = ["dog", "hound", "terrier", "spaniel", "retriever", "shepherd", 
                    "standard", "miniature", "toy", "giant", "mix"]
    words = name.split()
    # Keep at least the first word
    if len(words) > 1:
        # Get the core breed name (usually the first distinctive word)
        pass
    return name.strip()


def get_breed_base_name(breed_name: str) -> str:
    """Extract base breed name for fuzzy matching."""
    name = breed_name.lower().replace("_", " ").replace("-", " ")
    # Remove size qualifiers
    for qualifier in ["toy ", "miniature ", "standard ", "giant ", "mini "]:
        name = name.replace(qualifier, "")
    return name.strip()


def get_breed_info(breed_name: str) -> dict:
    """Get behavioral information for a breed."""
    if behavioral_data is None:
        return None
    
    # Normalize breed name for matching
    breed_lower = breed_name.lower().replace("_", " ").replace("-", " ")
    breed_base = get_breed_base_name(breed_name)
    
    # Try multiple matching strategies
    best_match = None
    best_score = 0
    
    for idx, row in behavioral_data.iterrows():
        name = str(row.get('Name', '')).lower().replace("_", " ").replace("-", " ")
        name_base = get_breed_base_name(name)
        
        score = 0
        
        # Exact match
        if breed_lower == name or breed_lower == name_base:
            score = 100
        # Breed name contains data name or vice versa
        elif breed_lower in name or name in breed_lower:
            score = 80
        # Base names match (e.g., "toy poodle" matches "poodle")
        elif breed_base in name_base or name_base in breed_base:
            score = 70
        # Check if main breed word matches
        else:
            breed_words = set(breed_lower.split())
            name_words = set(name.split())
            common = breed_words & name_words
            if common:
                # More common words = better match
                score = 50 + len(common) * 10
        
        if score > best_score:
            best_score = score
            best_match = row
    
    # Only use match if score is good enough
    if best_match is None or best_score < 50:
        return None
    
    row = best_match
    # Extract relevant information
    info = {}
    
    # Physical attributes - format ranges properly
    if 'min_ht (cm)' in row and pd.notna(row.get('min_ht (cm)')):
        min_ht = int(row.get('min_ht (cm)', 0))
        max_ht = int(row.get('max_ht (cm)', 0))
        if min_ht == max_ht or max_ht == 0:
            info['height'] = f"{min_ht} cm"
        else:
            info['height'] = f"{min_ht} - {max_ht} cm"
    
    if 'min_wt (kg)' in row and pd.notna(row.get('min_wt (kg)')):
        min_wt = int(row.get('min_wt (kg)', 0))
        max_wt = int(row.get('max_wt (kg)', 0))
        if min_wt == max_wt or max_wt == 0:
            info['weight'] = f"{min_wt} kg (breed avg)"
        else:
            info['weight'] = f"{min_wt} - {max_wt} kg (breed avg)"
    
    if 'min_life (yrs)' in row and pd.notna(row.get('min_life (yrs)')):
        min_life = int(row.get('min_life (yrs)', 0))
        max_life = int(row.get('max_life (yrs)', 0))
        if min_life == max_life or max_life == 0:
            info['lifespan'] = f"{min_life} years"
        else:
            info['lifespan'] = f"{min_life} - {max_life} years"
    
    # Behavioral traits (scale 1-5)
    traits = {}
    trait_mapping = {
        'Trainability.Level': 'Trainability',
        'Energy.Level': 'Energy',
        'Good.With.Young.Children': 'Good with Children',
        'Good.With.Other.Dogs': 'Good with Other Dogs',
        'Affectionate.With.Family': 'Affectionate',
        'Playfulness.Level': 'Playfulness',
        'Watchdog.Protective.Nature': 'Protective',
        'Openness.To.Strangers': 'Friendly to Strangers',
        'Barking.Level': 'Barking',
        'Mental.Stimulation.Needs': 'Mental Stimulation'
    }
    
    for col, label in trait_mapping.items():
        if col in row and pd.notna(row[col]):
            traits[label] = int(row[col])
    
    if traits:
        info['traits'] = traits
    
    # Additional info
    if 'temperament' in row and pd.notna(row['temperament']):
        info['temperament'] = row['temperament']
    if 'group' in row and pd.notna(row['group']):
        info['group'] = row['group']
    if 'description' in row and pd.notna(row['description']):
        # Truncate long descriptions
        desc = str(row['description'])
        info['description'] = desc[:500] + "..." if len(desc) > 500 else desc
    
    # Computed capability scores
    if traits:
        # Detection/Working Suitability
        work_traits = ['Trainability', 'Energy', 'Mental Stimulation', 'Protective']
        work_score = sum(traits.get(t, 3) for t in work_traits) / len(work_traits) * 20
        
        # Family Companion Score
        family_traits = ['Good with Children', 'Affectionate', 'Playfulness', 'Good with Other Dogs']
        family_score = sum(traits.get(t, 3) for t in family_traits) / len(family_traits) * 20
        
        # Guard Dog Potential
        guard_score = (traits.get('Protective', 3) + traits.get('Barking', 3) + 
                      (6 - traits.get('Friendly to Strangers', 3))) / 3 * 20
        
        info['capability_scores'] = {
            'Working Suitability': round(work_score),
            'Family Companion': round(family_score),
            'Guard Dog Potential': round(guard_score)
        }
    
    return info


# ============================================================================
# AUDIO MODEL (Cat vs Dog Classifier)
# ============================================================================

class AudioCNN(nn.Module):
    """4-block CNN for audio species classification."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(True), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(True), nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(True), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU(True),
            nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.gap(self.features(x)).flatten(1)
        return self.head(x)


def _build_audio_mel_fb(sr, n_fft, n_mels):
    """Build mel filterbank matrix."""
    fmax = sr / 2.0
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_pts = np.linspace(0, mel_max, n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i+1], bins[i+2]
        for j in range(l, c):
            if c != l: fb[i, j] = (j - l) / (c - l)
        for j in range(c, r):
            if r != c: fb[i, j] = (r - j) / (r - c)
    return fb


def _get_ffmpeg_path():
    """Find ffmpeg binary: check imageio_ffmpeg bundle, then system PATH."""
    # Try bundled ffmpeg from imageio_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    # Try well-known location in this venv
    venv_ffmpeg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               '.venv', 'Lib', 'site-packages', 'imageio_ffmpeg',
                               'binaries', 'ffmpeg-win-x86_64-v7.1.exe')
    if os.path.isfile(venv_ffmpeg):
        return venv_ffmpeg
    # Fallback: hope it's on system PATH
    return 'ffmpeg'

FFMPEG_PATH = _get_ffmpeg_path()


def convert_to_wav(filepath):
    """Convert any audio format to 16kHz mono WAV using ffmpeg. Returns path to WAV file."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.wav':
        return filepath  # Already WAV

    wav_path = filepath.rsplit('.', 1)[0] + '_converted.wav'
    try:
        import subprocess
        cmd = [
            FFMPEG_PATH, '-y', '-i', filepath,
            '-ar', '16000', '-ac', '1', '-sample_fmt', 's16',
            wav_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and os.path.isfile(wav_path):
            return wav_path
        else:
            print(f"ffmpeg conversion failed (code {result.returncode}): {result.stderr.decode(errors='ignore')}")
            return None
    except Exception as e:
        print(f"Audio conversion failed: {e}")
        return None


def load_audio_wav(filepath, sr=16000, duration=4):
    """Load wav, convert to mono float32, resample to target SR, pad/trim.
    Returns (waveform, error_msg). error_msg is None on success."""
    # Convert non-WAV formats first
    wav_path = convert_to_wav(filepath)
    if wav_path is None:
        return None, "Could not read audio file. Supported formats: WAV, MP3, OGG, FLAC, M4A."

    try:
        orig_sr, data = wavfile.read(wav_path)
    except Exception as e:
        return None, f"Could not read audio file: {str(e)}. Please upload a valid audio file."

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    if data.ndim == 2:
        data = data.mean(axis=1)

    if orig_sr != sr:
        num_samples = int(len(data) * sr / orig_sr)
        data = np.interp(np.linspace(0, len(data) - 1, num_samples),
                         np.arange(len(data)), data).astype(np.float32)

    target_len = sr * duration
    if len(data) < target_len:
        data = np.pad(data, (0, target_len - len(data)), mode="constant")
    else:
        start = (len(data) - target_len) // 2
        data = data[start:start + target_len]

    # Check if audio is essentially silent (failed read or empty file)
    if np.abs(data).max() < 1e-6:
        return None, "Audio file appears to be silent or empty. Please upload an audio file with audible cat/dog sounds."

    return data, None


def wav_to_mel(waveform, mel_fb, window, n_fft, hop_length):
    """Convert waveform to log-mel spectrogram."""
    n_frames = 1 + (len(waveform) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(n_frames, n_fft),
        strides=(waveform.strides[0] * hop_length, waveform.strides[0])
    ).copy()
    frames *= window
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    mel_spec = mel_fb @ spectrum.T
    return np.log(mel_spec + 1e-9)


def load_audio_model():
    """Load the trained audio CNN model."""
    global audio_model, audio_config, audio_mel_fb, audio_window, device

    model_path = os.path.join("outputs", "best_audio_model.pth")
    if not os.path.exists(model_path):
        print("Warning: No audio model found at outputs/best_audio_model.pth")
        return

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    audio_config = {
        "sample_rate": 16000,
        "duration": 4,
        "n_mels": 64,
        "n_fft": 1024,
        "hop_length": 512,
        "norm_mean": ckpt.get("norm_mean", -12.3881),
        "norm_std": ckpt.get("norm_std", 9.5806),
    }

    audio_model = AudioCNN(num_classes=2)
    audio_model.load_state_dict(ckpt["model_state_dict"])
    audio_model.to(device)
    audio_model.eval()

    audio_mel_fb = _build_audio_mel_fb(
        audio_config["sample_rate"], audio_config["n_fft"], audio_config["n_mels"]
    )
    audio_window = np.hanning(audio_config["n_fft"]).astype(np.float32)

    print(f"Audio model loaded: {sum(p.numel() for p in audio_model.parameters()):,} params")
    print(f"  Classes: {AUDIO_CLASS_NAMES}")
    print(f"  Val Accuracy: {ckpt.get('val_acc', 'N/A')}%")


def predict_audio(filepath):
    """Predict cat vs dog from audio file."""
    global audio_model, audio_config, audio_mel_fb, audio_window, device

    if audio_model is None:
        return {"error": "Audio model not loaded"}

    # Load and process audio
    waveform, load_error = load_audio_wav(
        filepath,
        sr=audio_config["sample_rate"],
        duration=audio_config["duration"]
    )

    if load_error is not None:
        return {"error": load_error}

    # Compute mel spectrogram
    mel = wav_to_mel(waveform, audio_mel_fb, audio_window,
                     audio_config["n_fft"], audio_config["hop_length"])

    if np.isnan(mel).any() or np.isinf(mel).any():
        return {"error": "Invalid audio - could not compute spectrogram"}

    # Normalize
    spec_tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,T)
    spec_tensor = (spec_tensor - audio_config["norm_mean"]) / (audio_config["norm_std"] + 1e-8)

    # Predict
    with torch.no_grad():
        spec_tensor = spec_tensor.to(device)
        logits = audio_model(spec_tensor)
        probs = F.softmax(logits, dim=1)[0]

    results = []
    for idx in range(len(AUDIO_CLASS_NAMES)):
        results.append({
            "class": AUDIO_CLASS_NAMES[idx],
            "label": AUDIO_LABELS[AUDIO_CLASS_NAMES[idx]],
            "confidence": float(probs[idx]) * 100
        })

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)

    # Add waveform data for visualization (downsample for frontend)
    waveform_viz = waveform[::160].tolist()  # ~100 points per second

    return {
        "predictions": results,
        "waveform": waveform_viz,
        "duration": audio_config["duration"],
        "sample_rate": audio_config["sample_rate"]
    }


def predict_breed(image_path: str, top_k: int = 5) -> list:
    """Predict breed from image."""
    global model, transform, device
    
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            breed_name = BREED_NAMES[idx].replace("_", " ").title()
            breed_info = get_breed_info(BREED_NAMES[idx])
            results.append({
                "breed": breed_name,
                "confidence": float(prob) * 100,
                "info": breed_info
            })
        
        return results
    except Exception as e:
        return [{"error": str(e)}]


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        predictions = predict_breed(filepath, top_k=5)
        
        return jsonify({
            "success": True,
            "image_path": f"/static/uploads/{filename}",
            "predictions": predictions
        })


@app.route('/predict_audio', methods=['POST'])
def predict_audio_route():
    """Handle audio upload and cat/dog prediction."""
    if 'file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_audio(filepath)
        
        if "error" in result:
            return jsonify({"success": False, "error": result["error"]}), 400
        
        return jsonify({
            "success": True,
            "audio_path": f"/static/uploads/{filename}",
            "predictions": result["predictions"],
            "waveform": result["waveform"],
            "duration": result["duration"],
            "sample_rate": result["sample_rate"]
        })


if __name__ == '__main__':
    print("=" * 50)
    print("🐕 Dog Breed Classification + Audio Classifier")
    print("=" * 50)
    load_model()
    load_behavioral_data()
    load_audio_model()
    print("\nStarting server at http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
