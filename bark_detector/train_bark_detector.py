"""
Train Bark Detector
=====================
Trains a binary CNN classifier (bark vs non-bark) using:
  - Positive samples: existing breed_audio WAV files (~27K dog barks)
  - Negative samples: cat audio from audio_organized + synthetic noise/silence

Output: outputs/bark_detector_best.pth

Usage:
    python -m bark_detector.train_bark_detector
    python bark_detector/train_bark_detector.py
"""

import os
import sys
import time
import json
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.io import wavfile
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import from our model module
sys.path.insert(0, str(Path(__file__).parent.parent))
from bark_detector.model import (
    BarkDetectorCNN, load_wav, wav_to_mel,
    SAMPLE_RATE, DURATION, N_MELS, N_FFT, HOP_LENGTH, DEVICE
)

# ============================================================================
# CONFIG
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Positive samples: dog barks from breed_audio
BARK_DIRS = [
    DATA_ROOT / "breed_audio" / "chihuahua",
    DATA_ROOT / "breed_audio" / "german_shepherd",
    DATA_ROOT / "breed_audio" / "husky",
    DATA_ROOT / "breed_audio" / "labrador",
    DATA_ROOT / "breed_audio" / "pitbull",
    DATA_ROOT / "breed_audio" / "shiba_inu",
]

# Negative samples: cat audio
CAT_DIRS = [
    DATA_ROOT / "audio_organized" / "train" / "cats",
    DATA_ROOT / "audio_organized" / "val" / "cats",
]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
SEED = 42
VAL_SPLIT = 0.15
MAX_POSITIVE = 7500    # ~1250 per breed (6 breeds), CPU-friendly
MAX_NEGATIVE = None     # Use all negatives available + synthetic

CLASS_NAMES = ["non_bark", "bark"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================================
# SYNTHETIC NEGATIVE GENERATION
# ============================================================================
def generate_silence(sr=SAMPLE_RATE, duration=DURATION):
    """Generate near-silent audio with tiny noise."""
    return np.random.randn(sr * duration).astype(np.float32) * 0.001


def generate_white_noise(sr=SAMPLE_RATE, duration=DURATION):
    """Generate white noise."""
    return np.random.randn(sr * duration).astype(np.float32) * 0.1


def generate_tone(freq=440, sr=SAMPLE_RATE, duration=DURATION):
    """Generate a pure tone (e.g., music note, alarm)."""
    t = np.linspace(0, duration, sr * duration, dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * freq * t)


def generate_speech_like(sr=SAMPLE_RATE, duration=DURATION):
    """Generate speech-like noise (modulated noise in speech frequency range)."""
    t = np.linspace(0, duration, sr * duration, dtype=np.float32)
    # Modulate noise with speech-like envelope
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # ~3 Hz modulation (syllable rate)
    noise = np.random.randn(sr * duration).astype(np.float32)
    
    # Bandpass-like filter: emphasize 100-4000 Hz (speech range)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1.0 / sr)
    mask = ((freqs >= 100) & (freqs <= 4000)).astype(np.float32)
    filtered = np.fft.irfft(spectrum * mask, n=len(noise)).astype(np.float32)
    
    return (filtered * envelope * 0.2).astype(np.float32)


def generate_environmental_noise(sr=SAMPLE_RATE, duration=DURATION):
    """Generate environmental noise (traffic-like low rumble + random clicks)."""
    t = np.linspace(0, duration, sr * duration, dtype=np.float32)
    
    # Low-frequency rumble
    rumble = 0.1 * np.sin(2 * np.pi * 80 * t) + 0.05 * np.sin(2 * np.pi * 120 * t)
    
    # Random clicks
    noise = np.random.randn(sr * duration).astype(np.float32) * 0.02
    clicks = np.zeros(sr * duration, dtype=np.float32)
    click_positions = random.sample(range(sr * duration), 20)
    for pos in click_positions:
        clicks[pos] = random.uniform(0.1, 0.3) * random.choice([-1, 1])
    
    return (rumble + noise + clicks).astype(np.float32)


SYNTHETIC_GENERATORS = [
    ("silence", generate_silence, 700),
    ("white_noise", generate_white_noise, 700),
    ("tone_440", lambda: generate_tone(440), 400),
    ("tone_1000", lambda: generate_tone(1000), 400),
    ("tone_2000", lambda: generate_tone(2000), 400),
    ("speech_like", generate_speech_like, 1200),
    ("env_noise", generate_environmental_noise, 1200),
]


# ============================================================================
# DATA COLLECTION
# ============================================================================
def collect_audio_files(dirs, extensions={".wav"}):
    """Collect all audio files from given directories."""
    files = []
    for d in dirs:
        if not d.exists():
            print(f"  WARNING: Directory not found: {d}")
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in extensions:
                files.append(f)
    return files


def collect_training_data():
    """
    Collect positive (bark) and negative (non-bark) training data.
    Balanced sampling: equal clips from each breed.
    
    Returns:
        positive_files: list of bark audio file paths
        negative_files: list of non-bark audio file paths
        synthetic_count: number of synthetic negatives to generate
    """
    print("\n  Collecting training data...")
    
    # Positives: dog barks — BALANCED per breed
    per_breed_cap = MAX_POSITIVE // len(BARK_DIRS)  # ~1250 per breed
    positive_files = []
    
    for breed_dir in BARK_DIRS:
        if not breed_dir.exists():
            print(f"  WARNING: {breed_dir} not found")
            continue
        breed_files = sorted([f for f in breed_dir.iterdir() if f.suffix.lower() == ".wav"])
        breed_name = breed_dir.name
        if len(breed_files) > per_breed_cap:
            sampled = random.sample(breed_files, per_breed_cap)
        else:
            sampled = breed_files
        positive_files.extend(sampled)
        print(f"    {breed_name}: {len(sampled)}/{len(breed_files)} selected")
    
    random.shuffle(positive_files)
    print(f"  Total bark samples: {len(positive_files)} (balanced ~{per_breed_cap}/breed)")
    
    # Negatives: cat audio
    negative_files = collect_audio_files(CAT_DIRS)
    print(f"  Cat audio files found: {len(negative_files)}")
    
    # Calculate synthetic count to balance — aim for ~1:1.2 ratio
    real_neg = len(negative_files)
    target_neg = int(len(positive_files) * 0.8)  # slightly fewer negatives is OK
    synthetic_needed = max(0, target_neg - real_neg)
    
    # Sum up synthetic generators
    total_synthetic = sum(count for _, _, count in SYNTHETIC_GENERATORS)
    synthetic_needed = min(synthetic_needed, total_synthetic)
    
    print(f"  Target ratio: ~1:0.8")
    print(f"  Real negatives: {real_neg}")
    print(f"  Synthetic negatives to generate: {synthetic_needed}")
    print(f"  Total positives: {len(positive_files)}")
    print(f"  Total negatives: ~{real_neg + synthetic_needed}")
    
    return positive_files, negative_files, synthetic_needed


# ============================================================================
# PRE-COMPUTE SPECTROGRAMS
# ============================================================================
def precompute_spectrograms(positive_files, negative_files, synthetic_count):
    """Load all audio, compute mel spectrograms, return tensors."""
    specs = []
    labels = []
    skipped = 0
    
    # Process positives (label = 1 = bark)
    print(f"\n  Processing {len(positive_files)} bark samples...")
    for i, fpath in enumerate(positive_files):
        wav = load_wav(fpath)
        mel = wav_to_mel(wav)
        if np.isnan(mel).any() or np.isinf(mel).any():
            skipped += 1
            continue
        specs.append(mel)
        labels.append(1)  # bark
        if (i + 1) % 2000 == 0:
            print(f"    Barks: {i+1}/{len(positive_files)}")
    
    print(f"    Barks done: {sum(1 for l in labels if l == 1)} loaded, {skipped} skipped")
    
    # Process negatives from files (label = 0 = non-bark)
    print(f"\n  Processing {len(negative_files)} non-bark audio files...")
    neg_skipped = 0
    for i, fpath in enumerate(negative_files):
        wav = load_wav(fpath)
        mel = wav_to_mel(wav)
        if np.isnan(mel).any() or np.isinf(mel).any():
            neg_skipped += 1
            continue
        specs.append(mel)
        labels.append(0)  # non-bark
        if (i + 1) % 200 == 0:
            print(f"    Non-bark files: {i+1}/{len(negative_files)}")
    
    print(f"    Non-bark files done: {sum(1 for l in labels if l == 0)} loaded, {neg_skipped} skipped")
    
    # Generate synthetic negatives
    print(f"\n  Generating synthetic non-bark samples...")
    remaining = synthetic_count
    for name, gen_fn, count in SYNTHETIC_GENERATORS:
        n = min(count, remaining)
        if n <= 0:
            break
        for _ in range(n):
            wav = gen_fn()
            mel = wav_to_mel(wav)
            if not (np.isnan(mel).any() or np.isinf(mel).any()):
                specs.append(mel)
                labels.append(0)
        remaining -= n
        print(f"    Generated {n} '{name}' samples")
    
    # Convert to tensors
    specs_t = torch.FloatTensor(np.stack(specs)).unsqueeze(1)  # (N, 1, n_mels, T)
    labels_t = torch.LongTensor(labels)
    
    n_bark = (labels_t == 1).sum().item()
    n_nonbark = (labels_t == 0).sum().item()
    print(f"\n  Total dataset: {len(labels)} samples")
    print(f"    Bark (positive):     {n_bark}")
    print(f"    Non-bark (negative): {n_nonbark}")
    print(f"    Tensor shape: {tuple(specs_t.shape)}")
    
    return specs_t, labels_t


# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================
def split_data(specs, labels, val_ratio=VAL_SPLIT):
    """Stratified train/val split."""
    bark_idx = (labels == 1).nonzero(as_tuple=True)[0].tolist()
    nonbark_idx = (labels == 0).nonzero(as_tuple=True)[0].tolist()
    
    random.shuffle(bark_idx)
    random.shuffle(nonbark_idx)
    
    val_bark = bark_idx[:int(len(bark_idx) * val_ratio)]
    val_nonbark = nonbark_idx[:int(len(nonbark_idx) * val_ratio)]
    train_bark = bark_idx[int(len(bark_idx) * val_ratio):]
    train_nonbark = nonbark_idx[int(len(nonbark_idx) * val_ratio):]
    
    train_idx = train_bark + train_nonbark
    val_idx = val_bark + val_nonbark
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    
    return (specs[train_idx], labels[train_idx],
            specs[val_idx], labels[val_idx])


# ============================================================================
# DATASET WITH AUGMENTATION
# ============================================================================
class BarkDataset(Dataset):
    def __init__(self, specs, labels, augment=False):
        self.specs = specs
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.specs[idx].clone()
        label = self.labels[idx]

        if self.augment:
            # Time masking
            if random.random() < 0.5:
                t = spec.shape[2]
                t_mask = random.randint(1, max(1, t // 8))
                t0 = random.randint(0, t - t_mask)
                spec[:, :, t0:t0 + t_mask] = 0
            # Frequency masking
            if random.random() < 0.5:
                f = spec.shape[1]
                f_mask = random.randint(1, max(1, f // 8))
                f0 = random.randint(0, f - f_mask)
                spec[:, f0:f0 + f_mask, :] = 0
            # Random gain
            if random.random() < 0.3:
                spec = spec + random.uniform(-0.3, 0.3)

        return spec, label


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    preds_all, targets_all = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        p = out.argmax(1)
        correct += (p == y).sum().item()
        total += y.size(0)
        preds_all.extend(p.cpu().numpy())
        targets_all.extend(y.cpu().numpy())
    return loss_sum / total, 100.0 * correct / total, np.array(preds_all), np.array(targets_all)


# ============================================================================
# PLOTS
# ============================================================================
def save_training_plots(tl, vl, ta, va, preds, targets, output_dir):
    """Save training curves and confusion matrix."""
    # Training curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(tl) + 1)
    a1.plot(ep, tl, "b-o", ms=3, label="Train")
    a1.plot(ep, vl, "r-o", ms=3, label="Val")
    a1.set(xlabel="Epoch", ylabel="Loss", title="Bark Detector - Loss")
    a1.legend()
    a1.grid(True, alpha=0.3)
    a2.plot(ep, ta, "b-o", ms=3, label="Train")
    a2.plot(ep, va, "r-o", ms=3, label="Val")
    a2.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Bark Detector - Accuracy")
    a2.legend()
    a2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "bark_detector_curves.png", dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True", title="Bark Detector - Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=18)
    plt.tight_layout()
    plt.savefig(output_dir / "bark_detector_confusion.png", dpi=150)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("  BARK DETECTOR - Training Pipeline")
    print("  Binary CNN: bark (dog) vs non-bark (cat/noise/silence)")
    print("=" * 70)
    print(f"\n  Device: {DEVICE}")
    print(f"  SR: {SAMPLE_RATE}Hz | Clip: {DURATION}s | Mels: {N_MELS}")
    print(f"  Batch: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LR}")

    # ---- Phase 1: Collect data ----
    print(f"\n{'─' * 50}")
    print("  Phase 1: Data Collection")
    print(f"{'─' * 50}")
    positive_files, negative_files, synthetic_count = collect_training_data()

    # ---- Phase 2: Pre-compute spectrograms ----
    print(f"\n{'─' * 50}")
    print("  Phase 2: Computing Spectrograms")
    print(f"{'─' * 50}")
    t0 = time.time()
    all_specs, all_labels = precompute_spectrograms(
        positive_files, negative_files, synthetic_count
    )
    print(f"  Complete in {time.time() - t0:.1f}s")

    # ---- Phase 3: Split ----
    print(f"\n{'─' * 50}")
    print("  Phase 3: Train/Val Split")
    print(f"{'─' * 50}")
    train_specs, train_labels, val_specs, val_labels = split_data(all_specs, all_labels)

    print(f"  Train: {len(train_labels)} "
          f"(bark={int((train_labels == 1).sum())}, "
          f"non_bark={int((train_labels == 0).sum())})")
    print(f"  Val:   {len(val_labels)} "
          f"(bark={int((val_labels == 1).sum())}, "
          f"non_bark={int((val_labels == 0).sum())})")

    # Normalize
    mean = train_specs.mean()
    std = train_specs.std()
    train_specs = (train_specs - mean) / (std + 1e-8)
    val_specs = (val_specs - mean) / (std + 1e-8)
    print(f"  Normalized: mean={mean:.4f}, std={std:.4f}")

    # DataLoaders
    train_loader = DataLoader(
        BarkDataset(train_specs, train_labels, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        BarkDataset(val_specs, val_labels, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Class weights
    n_nonbark = (train_labels == 0).sum().item()
    n_bark = (train_labels == 1).sum().item()
    total = len(train_labels)
    weights = torch.FloatTensor([total / (2 * n_nonbark), total / (2 * n_bark)]).to(DEVICE)
    print(f"  Class weights: non_bark={weights[0]:.3f}, bark={weights[1]:.3f}")

    # ---- Phase 4: Model ----
    print(f"\n{'─' * 50}")
    print("  Phase 4: Model Setup")
    print(f"{'─' * 50}")
    model = BarkDetectorCNN(num_classes=2).to(DEVICE)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  BarkDetectorCNN: {nparams:,} params")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # ---- Phase 5: Training ----
    print(f"\n{'=' * 70}")
    print("  Phase 5: TRAINING")
    print(f"{'=' * 70}")

    best_acc, best_ep = 0.0, 0
    tl_hist, vl_hist, ta_hist, va_hist = [], [], [], []
    patience_ctr = 0
    PATIENCE = 6

    t_start = time.time()
    for ep in range(1, EPOCHS + 1):
        t_ep = time.time()
        tr_l, tr_a = train_epoch(model, train_loader, criterion, optimizer)
        vl_l, vl_a, vl_p, vl_t = evaluate(model, val_loader, criterion)
        scheduler.step(vl_l)

        tl_hist.append(tr_l)
        vl_hist.append(vl_l)
        ta_hist.append(tr_a)
        va_hist.append(vl_a)
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t_ep

        flag = ""
        if vl_a > best_acc:
            best_acc, best_ep = vl_a, ep
            patience_ctr = 0
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_acc": vl_a,
                "val_loss": vl_l,
                "class_names": CLASS_NAMES,
                "norm_mean": mean.item(),
                "norm_std": std.item(),
            }, OUTPUT_DIR / "bark_detector_best.pth")
            flag = " ** BEST **"
        else:
            patience_ctr += 1

        print(f"  Ep {ep:02d}/{EPOCHS} | "
              f"TrL {tr_l:.4f} TrA {tr_a:5.1f}% | "
              f"VlL {vl_l:.4f} VlA {vl_a:5.1f}% | "
              f"LR {lr_now:.1e} | {dt:.1f}s{flag}")

        if patience_ctr >= PATIENCE:
            print(f"\n  Early stopping (no gain for {PATIENCE} epochs)")
            break

    total_t = time.time() - t_start
    print(f"\n  Done in {total_t:.1f}s ({total_t / 60:.1f} min)")
    print(f"  Best: {best_acc:.1f}% @ epoch {best_ep}")

    # ---- Phase 6: Final Evaluation ----
    print(f"\n{'=' * 70}")
    print("  Phase 6: EVALUATION")
    print(f"{'=' * 70}")

    ckpt = torch.load(OUTPUT_DIR / "bark_detector_best.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    _, acc, preds, targets = evaluate(model, val_loader, criterion)

    print(f"\n  Val Accuracy: {acc:.2f}%\n")
    report = classification_report(targets, preds, target_names=CLASS_NAMES, digits=4)
    for line in report.split("\n"):
        print(f"  {line}")

    # ---- Phase 7: Save ----
    save_training_plots(tl_hist, vl_hist, ta_hist, va_hist, preds, targets, OUTPUT_DIR)
    print(f"\n  Saved: bark_detector_curves.png")
    print(f"  Saved: bark_detector_confusion.png")

    summary = {
        "model": "BarkDetectorCNN",
        "task": "bark_vs_non_bark",
        "classes": CLASS_NAMES,
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "best_epoch": best_ep,
        "best_val_acc": round(best_acc, 2),
        "total_params": nparams,
        "training_time_sec": round(total_t, 1),
        "norm_mean": round(mean.item(), 6),
        "norm_std": round(std.item(), 6),
        "config": {
            "sample_rate": SAMPLE_RATE,
            "duration": DURATION,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
        },
    }
    with open(OUTPUT_DIR / "bark_detector_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  BARK DETECTOR TRAINING COMPLETE!")
    print(f"  Model:     outputs/bark_detector_best.pth")
    print(f"  Curves:    outputs/bark_detector_curves.png")
    print(f"  Matrix:    outputs/bark_detector_confusion.png")
    print(f"  Summary:   outputs/bark_detector_summary.json")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
