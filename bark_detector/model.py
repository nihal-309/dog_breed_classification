"""
Bark Detector CNN Model
========================
Binary classifier: bark (1) vs non-bark (0)
Uses mel spectrogram input, same preprocessing as train_audio.py.

Architecture: 4-block CNN with Global Average Pooling
Input: (B, 1, n_mels, T) mel spectrogram
Output: (B, 2) logits [non_bark, bark]
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.io import wavfile

# ============================================================================
# AUDIO CONFIG (matches train_audio.py)
# ============================================================================
SAMPLE_RATE = 16000
DURATION = 4          # seconds
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# AUDIO PROCESSING (reused from train_audio.py)
# ============================================================================
def load_wav(filepath, sr=SAMPLE_RATE, duration=DURATION):
    """Load wav, convert to mono float32, resample to target SR, pad/trim."""
    try:
        orig_sr, data = wavfile.read(str(filepath))
    except Exception:
        return np.zeros(sr * duration, dtype=np.float32)

    # Normalize to float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    # Mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Resample if needed
    if orig_sr != sr:
        num_samples = int(len(data) * sr / orig_sr)
        data = np.interp(
            np.linspace(0, len(data) - 1, num_samples),
            np.arange(len(data)), data
        ).astype(np.float32)

    # Pad or center-crop to target length
    target_len = sr * duration
    if len(data) < target_len:
        data = np.pad(data, (0, target_len - len(data)), mode="constant")
    else:
        start = (len(data) - target_len) // 2
        data = data[start:start + target_len]

    return data


def _build_mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SAMPLE_RATE):
    """Build mel filterbank matrix."""
    fmax = sr / 2.0
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_pts = np.linspace(0, mel_max, n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        for j in range(l, c):
            if c != l:
                fb[i, j] = (j - l) / (c - l)
        for j in range(c, r):
            if r != c:
                fb[i, j] = (r - j) / (r - c)
    return fb


# Pre-compute once on import
MEL_FB = _build_mel_filterbank()
WINDOW = np.hanning(N_FFT).astype(np.float32)


def wav_to_mel(waveform):
    """Waveform (float32 array) -> log-mel spectrogram (n_mels, time_frames)."""
    n_frames = 1 + (len(waveform) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(n_frames, N_FFT),
        strides=(waveform.strides[0] * HOP_LENGTH, waveform.strides[0])
    ).copy()
    frames *= WINDOW
    spectrum = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2
    mel_spec = MEL_FB @ spectrum.T
    return np.log(mel_spec + 1e-9)


# ============================================================================
# BARK DETECTOR CNN
# ============================================================================
class BarkDetectorCNN(nn.Module):
    """
    Binary CNN classifier for bark detection.
    4-block architecture with Global Average Pooling.
    
    Input:  (B, 1, 64, T) mel spectrogram
    Output: (B, 2) logits [non_bark, bark]
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 32
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.head(x)


# ============================================================================
# INFERENCE HELPER
# ============================================================================
class BarkDetector:
    """
    High-level bark detector for inference.
    
    Usage:
        detector = BarkDetector("outputs/bark_detector_best.pth")
        is_bark, confidence = detector.predict("audio.wav")
        # is_bark: True/False
        # confidence: 0.0 - 1.0
    """
    def __init__(self, model_path, threshold=0.5):
        self.threshold = threshold
        self.device = DEVICE

        # Load checkpoint
        ckpt = torch.load(str(model_path), map_location=self.device, weights_only=True)
        self.norm_mean = ckpt.get("norm_mean", 0.0)
        self.norm_std = ckpt.get("norm_std", 1.0)

        self.model = BarkDetectorCNN(num_classes=2)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, audio_path):
        """
        Predict whether an audio file contains a bark.
        
        Returns:
            (is_bark: bool, confidence: float)
        """
        wav = load_wav(str(audio_path))
        mel = wav_to_mel(wav)

        if np.isnan(mel).any() or np.isinf(mel).any():
            return False, 0.0

        # Normalize using training stats
        mel = (mel - self.norm_mean) / (self.norm_std + 1e-8)

        # To tensor: (1, 1, n_mels, T)
        tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        bark_prob = probs[0, 1].item()  # class 1 = bark

        return bark_prob >= self.threshold, bark_prob

    @torch.no_grad()
    def predict_batch(self, audio_paths, batch_size=32):
        """
        Predict bark/non-bark for a batch of audio files.
        
        Returns:
            list of (is_bark: bool, confidence: float)
        """
        results = []
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            mels = []
            valid_indices = []

            for j, path in enumerate(batch_paths):
                wav = load_wav(str(path))
                mel = wav_to_mel(wav)
                if np.isnan(mel).any() or np.isinf(mel).any():
                    results.append((False, 0.0))
                else:
                    mel = (mel - self.norm_mean) / (self.norm_std + 1e-8)
                    mels.append(mel)
                    valid_indices.append(i + j)

            if mels:
                tensor = torch.FloatTensor(np.stack(mels)).unsqueeze(1).to(self.device)
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)

                mel_idx = 0
                for j in range(len(batch_paths)):
                    if (i + j) in valid_indices:
                        bark_prob = probs[mel_idx, 1].item()
                        results.append((bark_prob >= self.threshold, bark_prob))
                        mel_idx += 1

        return results

    @torch.no_grad()
    def predict_waveform(self, waveform_np):
        """
        Predict from a numpy waveform array (already loaded).
        
        Args:
            waveform_np: float32 numpy array, mono, at SAMPLE_RATE
            
        Returns:
            (is_bark: bool, confidence: float)
        """
        # Pad/trim to expected length
        target_len = SAMPLE_RATE * DURATION
        if len(waveform_np) < target_len:
            waveform_np = np.pad(waveform_np, (0, target_len - len(waveform_np)))
        else:
            start = (len(waveform_np) - target_len) // 2
            waveform_np = waveform_np[start:start + target_len]

        mel = wav_to_mel(waveform_np.astype(np.float32))
        if np.isnan(mel).any() or np.isinf(mel).any():
            return False, 0.0

        mel = (mel - self.norm_mean) / (self.norm_std + 1e-8)
        tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        bark_prob = probs[0, 1].item()

        return bark_prob >= self.threshold, bark_prob
