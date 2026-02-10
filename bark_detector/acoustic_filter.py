"""
Acoustic Heuristic Filter
===========================
Layer 2 of the bark detection pipeline.
Uses acoustic features (spectral centroid, onset rate, ZCR, energy)
to quickly filter audio that is unlikely to contain barks.

This is a FAST pre-filter that runs before the CNN model,
rejecting obvious non-bark audio (silence, music, speech, noise).
"""

import numpy as np
from scipy.io import wavfile
from pathlib import Path

# Reuse audio config from model
SAMPLE_RATE = 16000
DURATION = 4


def load_wav_for_analysis(filepath, sr=SAMPLE_RATE, duration=DURATION):
    """Load wav file for acoustic analysis."""
    try:
        orig_sr, data = wavfile.read(str(filepath))
    except Exception:
        return None

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    if data.ndim == 2:
        data = data.mean(axis=1)

    if orig_sr != sr:
        num_samples = int(len(data) * sr / orig_sr)
        data = np.interp(
            np.linspace(0, len(data) - 1, num_samples),
            np.arange(len(data)), data
        ).astype(np.float32)

    target_len = sr * duration
    if len(data) < target_len:
        data = np.pad(data, (0, target_len - len(data)), mode="constant")
    else:
        start = (len(data) - target_len) // 2
        data = data[start:start + target_len]

    return data


# ============================================================================
# ACOUSTIC FEATURES
# ============================================================================

def compute_spectral_centroid(waveform, sr=SAMPLE_RATE, n_fft=1024, hop_length=512):
    """
    Compute spectral centroid (center of mass of the spectrum).
    Dog barks typically have centroid in 500-2500 Hz range.
    """
    n_frames = 1 + (len(waveform) - n_fft) // hop_length
    if n_frames <= 0:
        return 0.0

    window = np.hanning(n_fft).astype(np.float32)
    centroids = []

    for i in range(n_frames):
        start = i * hop_length
        frame = waveform[start:start + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        total_energy = spectrum.sum()
        if total_energy > 1e-10:
            centroid = (freqs * spectrum).sum() / total_energy
            centroids.append(centroid)

    return np.mean(centroids) if centroids else 0.0


def compute_onset_rate(waveform, sr=SAMPLE_RATE, hop_length=512, threshold=0.3):
    """
    Detect onsets (sudden energy changes) using spectral flux.
    Dog barks are impulsive -> expect 2-20 onsets in a 4-second clip.
    """
    n_fft = 1024
    n_frames = 1 + (len(waveform) - n_fft) // hop_length
    if n_frames <= 1:
        return 0

    window = np.hanning(n_fft).astype(np.float32)
    magnitudes = []

    for i in range(n_frames):
        start = i * hop_length
        frame = waveform[start:start + n_fft] * window
        magnitudes.append(np.abs(np.fft.rfft(frame)))

    magnitudes = np.array(magnitudes)

    # Spectral flux: sum of positive differences between consecutive frames
    flux = np.zeros(n_frames - 1)
    for i in range(1, n_frames):
        diff = magnitudes[i] - magnitudes[i - 1]
        flux[i - 1] = np.sum(np.maximum(0, diff))

    # Normalize
    if flux.max() > 0:
        flux = flux / flux.max()

    # Count onsets above threshold
    onsets = 0
    in_onset = False
    for val in flux:
        if val > threshold and not in_onset:
            onsets += 1
            in_onset = True
        elif val <= threshold * 0.5:
            in_onset = False

    return onsets


def compute_zcr(waveform):
    """
    Compute Zero Crossing Rate.
    Dog barks typically have ZCR in 0.02-0.15 range.
    Pure tones have very low ZCR, noise has very high ZCR.
    """
    if len(waveform) < 2:
        return 0.0
    signs = np.sign(waveform)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return crossings / len(waveform)


def compute_rms_energy(waveform, frame_length=1024, hop_length=512):
    """
    Compute RMS energy per frame and return statistics.
    Barks have high energy variation (loud bursts + quiet gaps).
    """
    n_frames = 1 + (len(waveform) - frame_length) // hop_length
    if n_frames <= 0:
        return 0.0, 0.0, 0.0

    energies = []
    for i in range(n_frames):
        start = i * hop_length
        frame = waveform[start:start + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))
        energies.append(rms)

    energies = np.array(energies)
    return energies.mean(), energies.std(), energies.max()


def compute_energy_variation(waveform, frame_length=1024, hop_length=512):
    """
    Ratio of energy std to mean. Barks have high variation
    (intermittent bursts), while continuous sounds (music, fan) have low.
    """
    mean_e, std_e, max_e = compute_rms_energy(waveform, frame_length, hop_length)
    if mean_e < 1e-8:
        return 0.0
    return std_e / mean_e


def compute_dominant_frequency(waveform, sr=SAMPLE_RATE, n_fft=4096):
    """
    Find the dominant frequency in the audio.
    Dog barks typically peak between 300-3000 Hz.
    """
    spectrum = np.abs(np.fft.rfft(waveform, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    dominant_idx = np.argmax(spectrum)
    return freqs[dominant_idx]


# ============================================================================
# BARK LIKELIHOOD SCORING
# ============================================================================

# Thresholds derived from analysis of known bark audio
BARK_THRESHOLDS = {
    "spectral_centroid": (300, 3500),    # Hz - bark frequency range
    "onset_rate": (1, 30),                # number of onsets in 4s clip
    "zcr": (0.01, 0.20),                 # zero crossing rate
    "energy_variation": (0.3, float("inf")),  # barks are bursty
    "rms_mean_min": 0.005,                # not silence
    "dominant_freq": (200, 4000),          # Hz
}


def compute_bark_score(waveform, sr=SAMPLE_RATE):
    """
    Compute a heuristic bark likelihood score (0.0 - 1.0).
    
    Combines multiple acoustic features into a single score.
    Higher score = more likely to be a bark.
    
    Args:
        waveform: float32 numpy array (mono, at SAMPLE_RATE)
        sr: sample rate
        
    Returns:
        dict with:
            - score: float (0.0 - 1.0)
            - features: dict of computed features
            - reasons: list of reasons for rejection/acceptance
    """
    features = {}
    reasons = []
    score_components = []

    # 1. Check for silence
    rms_mean, rms_std, rms_max = compute_rms_energy(waveform)
    features["rms_mean"] = rms_mean
    features["rms_std"] = rms_std
    features["rms_max"] = rms_max

    if rms_mean < BARK_THRESHOLDS["rms_mean_min"]:
        return {"score": 0.0, "features": features, "reasons": ["Audio is near-silent"]}

    # 2. Spectral centroid
    centroid = compute_spectral_centroid(waveform, sr)
    features["spectral_centroid"] = centroid
    lo, hi = BARK_THRESHOLDS["spectral_centroid"]
    if lo <= centroid <= hi:
        score_components.append(1.0)
        reasons.append(f"Spectral centroid ({centroid:.0f} Hz) in bark range")
    else:
        # Partial credit if close
        if centroid < lo:
            penalty = max(0, 1 - (lo - centroid) / lo)
        else:
            penalty = max(0, 1 - (centroid - hi) / hi)
        score_components.append(penalty * 0.5)
        reasons.append(f"Spectral centroid ({centroid:.0f} Hz) outside bark range ({lo}-{hi} Hz)")

    # 3. Onset rate
    onsets = compute_onset_rate(waveform, sr)
    features["onset_rate"] = onsets
    lo, hi = BARK_THRESHOLDS["onset_rate"]
    if lo <= onsets <= hi:
        score_components.append(1.0)
        reasons.append(f"Onset rate ({onsets}) consistent with barking")
    elif onsets == 0:
        score_components.append(0.2)
        reasons.append(f"No onsets detected — unlikely bark")
    else:
        score_components.append(0.4)
        reasons.append(f"Onset rate ({onsets}) outside typical bark range ({lo}-{hi})")

    # 4. Zero Crossing Rate
    zcr = compute_zcr(waveform)
    features["zcr"] = zcr
    lo, hi = BARK_THRESHOLDS["zcr"]
    if lo <= zcr <= hi:
        score_components.append(1.0)
        reasons.append(f"ZCR ({zcr:.4f}) in bark range")
    else:
        score_components.append(0.3)
        reasons.append(f"ZCR ({zcr:.4f}) outside bark range ({lo}-{hi})")

    # 5. Energy variation (burstiness)
    energy_var = compute_energy_variation(waveform)
    features["energy_variation"] = energy_var
    min_var = BARK_THRESHOLDS["energy_variation"][0]
    if energy_var >= min_var:
        score_components.append(1.0)
        reasons.append(f"Energy variation ({energy_var:.3f}) shows bursts (barky)")
    else:
        score_components.append(0.4)
        reasons.append(f"Energy variation ({energy_var:.3f}) too uniform — may be continuous sound")

    # 6. Dominant frequency
    dom_freq = compute_dominant_frequency(waveform, sr)
    features["dominant_freq"] = dom_freq
    lo, hi = BARK_THRESHOLDS["dominant_freq"]
    if lo <= dom_freq <= hi:
        score_components.append(1.0)
        reasons.append(f"Dominant freq ({dom_freq:.0f} Hz) in bark range")
    else:
        score_components.append(0.3)
        reasons.append(f"Dominant freq ({dom_freq:.0f} Hz) outside bark range ({lo}-{hi} Hz)")

    # Weighted average
    weights = [0.20, 0.20, 0.15, 0.20, 0.25]  # centroid, onset, zcr, energy_var, dom_freq
    final_score = sum(w * s for w, s in zip(weights, score_components))

    return {
        "score": min(1.0, max(0.0, final_score)),
        "features": features,
        "reasons": reasons,
    }


def is_likely_bark(filepath_or_waveform, threshold=0.5, sr=SAMPLE_RATE):
    """
    Quick check: is this audio likely a bark?
    
    Args:
        filepath_or_waveform: Path to WAV file or numpy waveform array
        threshold: minimum score to accept (default 0.5)
        sr: sample rate (if waveform provided)
        
    Returns:
        (is_bark: bool, score: float, details: dict)
    """
    if isinstance(filepath_or_waveform, (str, Path)):
        waveform = load_wav_for_analysis(str(filepath_or_waveform), sr)
        if waveform is None:
            return False, 0.0, {"error": "Failed to load audio"}
    else:
        waveform = filepath_or_waveform

    result = compute_bark_score(waveform, sr)
    return result["score"] >= threshold, result["score"], result


def batch_filter(audio_paths, threshold=0.5, verbose=False):
    """
    Filter a batch of audio files, returning only those likely containing barks.
    
    Args:
        audio_paths: list of Path objects or strings
        threshold: minimum bark score
        verbose: print progress
        
    Returns:
        (passed: list of paths, rejected: list of (path, score, reasons))
    """
    passed = []
    rejected = []

    for i, path in enumerate(audio_paths):
        is_bark, score, details = is_likely_bark(path, threshold)
        if is_bark:
            passed.append(path)
        else:
            rejected.append((path, score, details.get("reasons", [])))

        if verbose and (i + 1) % 50 == 0:
            print(f"  Acoustic filter: {i+1}/{len(audio_paths)} "
                  f"| passed: {len(passed)} | rejected: {len(rejected)}")

    if verbose:
        print(f"  Acoustic filter complete: {len(passed)}/{len(audio_paths)} passed "
              f"(threshold={threshold})")

    return passed, rejected


# ============================================================================
# CLI TEST
# ============================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python acoustic_filter.py <audio_file.wav> [threshold]")
        sys.exit(1)

    filepath = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    is_bark, score, details = is_likely_bark(filepath, threshold)

    print(f"\nFile: {filepath}")
    print(f"Bark Score: {score:.3f} (threshold: {threshold})")
    print(f"Verdict: {'BARK' if is_bark else 'NOT BARK'}")
    print(f"\nFeatures:")
    for k, v in details.get("features", {}).items():
        print(f"  {k}: {v:.4f}")
    print(f"\nReasons:")
    for r in details.get("reasons", []):
        print(f"  - {r}")
