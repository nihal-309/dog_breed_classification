"""
Dog Breed Audio Classifier - Training Script
===============================================
Trains a CNN to classify dog breeds from bark audio (mel spectrograms).
Pre-computes mel spectrograms into RAM, then trains with standard PyTorch loop.

Reads from: data/breed_audio/<breed_name>/*.wav
Outputs:     outputs/breed_classifier_best.pth
             outputs/breed_classifier_metrics.json
             outputs/breed_classifier_confusion.png

Usage:
    python train_breed_classifier.py
    python train_breed_classifier.py --max-per-breed 500
    python train_breed_classifier.py --epochs 20 --batch-size 64
    python train_breed_classifier.py --min-samples 10  # skip breeds with < 10 clips
"""

import os
import sys
import json
import time
import random
import argparse
import warnings
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.io import wavfile
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIG
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data" / "breed_audio"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 4          # seconds
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

DEVICE = torch.device("cpu")  # CPU-only training


# ============================================================================
# AUDIO PROCESSING (same as bark_detector/model.py)
# ============================================================================
def _build_mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SAMPLE_RATE):
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


MEL_FB = _build_mel_filterbank()
WINDOW = np.hanning(N_FFT).astype(np.float32)


def load_wav(filepath, sr=SAMPLE_RATE, duration=DURATION):
    """Load wav -> mono float32, resample, pad/trim to fixed length."""
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

    # Skip near-silent clips
    rms = np.sqrt(np.mean(data ** 2))
    if rms < 0.005:
        return None

    return data


def wav_to_mel(waveform):
    """Waveform -> log-mel spectrogram (n_mels, time_frames)."""
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
# DATA AUGMENTATION (on mel spectrograms)
# ============================================================================
def augment_mel(mel):
    """Apply random augmentations to mel spectrogram for training."""
    mel = mel.copy()

    # Time masking (mask a random time segment)
    if random.random() < 0.5:
        t_len = mel.shape[1]
        mask_len = random.randint(1, max(1, t_len // 8))
        start = random.randint(0, t_len - mask_len)
        mel[:, start:start + mask_len] = mel.mean()

    # Frequency masking (mask random frequency bands)
    if random.random() < 0.5:
        f_len = mel.shape[0]
        mask_len = random.randint(1, max(1, f_len // 8))
        start = random.randint(0, f_len - mask_len)
        mel[start:start + mask_len, :] = mel.mean()

    # Random gain
    if random.random() < 0.3:
        gain = random.uniform(0.8, 1.2)
        mel = mel * gain

    # Add slight noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.05, mel.shape).astype(np.float32)
        mel = mel + noise

    return mel


# ============================================================================
# DATASET
# ============================================================================
class BreedAudioDataset(Dataset):
    """Pre-computed mel spectrogram dataset for breed classification."""

    def __init__(self, mels, labels, augment=False):
        self.mels = mels        # list of numpy arrays (n_mels, T)
        self.labels = labels    # list of int class indices
        self.augment = augment

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        mel = self.mels[idx]

        if self.augment:
            mel = augment_mel(mel)

        # (1, n_mels, T) tensor
        x = torch.from_numpy(mel).unsqueeze(0)
        y = self.labels[idx]
        return x, y


# ============================================================================
# MODEL: Breed Classifier CNN
# ============================================================================
class BreedClassifierCNN(nn.Module):
    """
    Multi-class CNN for breed classification from mel spectrograms.
    5-block architecture with Global Average Pooling.
    
    Deeper than BarkDetectorCNN since this is a harder problem
    (74 classes vs 2).

    Input:  (B, 1, 64, T) mel spectrogram
    Output: (B, num_classes) logits
    """
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 32
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 5: 256 -> 512
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.head(x)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_breed_data(data_root, max_per_breed=700, min_samples=10, val_split=0.2):
    """
    Scan breed_audio/ directories, load WAVs, compute mel spectrograms.
    
    Args:
        data_root: path to data/breed_audio/
        max_per_breed: max clips per breed (caps at 700)
        min_samples: skip breeds with fewer clips than this
        val_split: fraction for validation
        
    Returns:
        train_dataset, val_dataset, class_names, class_counts
    """
    data_root = Path(data_root)
    
    # Discover breeds
    breed_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    print(f"\n{'=' * 60}")
    print(f"  LOADING BREED AUDIO DATA")
    print(f"  Source: {data_root}")
    print(f"  Max per breed: {max_per_breed}")
    print(f"  Min samples: {min_samples}")
    print(f"{'=' * 60}\n")
    
    all_mels = []
    all_labels = []
    class_names = []
    class_counts = {}
    skipped_breeds = []
    
    for breed_dir in breed_dirs:
        breed = breed_dir.name
        wav_files = sorted([f for f in breed_dir.iterdir() 
                           if f.suffix.lower() == ".wav"])
        
        if len(wav_files) < min_samples:
            skipped_breeds.append((breed, len(wav_files)))
            continue
        
        # Sample if too many
        if len(wav_files) > max_per_breed:
            wav_files = random.sample(wav_files, max_per_breed)
        
        class_idx = len(class_names)
        class_names.append(breed)
        loaded = 0
        
        for wf in wav_files:
            wav = load_wav(str(wf))
            if wav is None:
                continue
            mel = wav_to_mel(wav)
            all_mels.append(mel)
            all_labels.append(class_idx)
            loaded += 1
        
        class_counts[breed] = loaded
        print(f"  [{class_idx + 1:3d}] {breed:35s} -> {loaded:4d} clips"
              f" (from {len([f for f in breed_dir.iterdir() if f.suffix.lower() == '.wav'])} total)")
    
    if skipped_breeds:
        print(f"\n  Skipped {len(skipped_breeds)} breeds (< {min_samples} clips):")
        for name, count in skipped_breeds:
            print(f"    - {name}: {count} clips")
    
    # Shuffle and split
    total = len(all_mels)
    indices = list(range(total))
    random.shuffle(indices)
    
    val_size = int(total * val_split)
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]
    
    train_mels = [all_mels[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_mels = [all_mels[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]
    
    train_ds = BreedAudioDataset(train_mels, train_labels, augment=True)
    val_ds = BreedAudioDataset(val_mels, val_labels, augment=False)
    
    print(f"\n  Total breeds: {len(class_names)}")
    print(f"  Total clips:  {total}")
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")
    
    return train_ds, val_ds, class_names, class_counts


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_model(train_ds, val_ds, num_classes, epochs=25, batch_size=32, lr=1e-3):
    """Train the breed classifier CNN."""
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    
    model = BreedClassifierCNN(num_classes=num_classes).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: BreedClassifierCNN")
    print(f"  Parameters: {total_params:,}")
    print(f"  Classes: {num_classes}")
    print(f"  Device: {DEVICE}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {epochs}")
    
    # Class-weighted loss to handle imbalance
    label_counts = Counter(train_ds.labels)
    total_samples = len(train_ds)
    weights = torch.zeros(num_classes)
    for cls_idx, count in label_counts.items():
        weights[cls_idx] = total_samples / (num_classes * count)
    # Clamp weights to avoid extreme values
    weights = torch.clamp(weights, 0.5, 5.0)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print(f"\n{'=' * 70}")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'LR':>10} | {'Time':>6}")
    print(f"{'=' * 70}")
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, preds = logits.max(1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        
        train_loss = running_loss / total
        train_acc = correct / total * 100
        
        # ---- VALIDATE ----
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                val_loss_sum += loss.item() * batch_x.size(0)
                _, preds = logits.max(1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(batch_y.cpu().numpy())
        
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total * 100
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Save best model
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "val_acc": val_acc,
                "epoch": epoch,
            }, str(OUTPUT_DIR / "breed_classifier_best.pth"))
            marker = " *"
        
        print(f"  {epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | "
              f"{val_loss:10.4f} | {val_acc:8.2f}% | {current_lr:10.7f} | "
              f"{epoch_time:5.1f}s{marker}")
    
    total_time = time.time() - start_time
    print(f"{'=' * 70}")
    print(f"  Training complete in {total_time / 60:.1f} minutes")
    print(f"  Best val accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    
    return model, history, best_val_acc, best_epoch, all_preds, all_true


# ============================================================================
# EVALUATION & PLOTS
# ============================================================================
def plot_training_curves(history):
    """Plot loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train")
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "breed_classifier_curves.png"), dpi=150)
    plt.close()
    print(f"  Saved: outputs/breed_classifier_curves.png")


def compute_per_class_accuracy(all_true, all_preds, class_names):
    """Compute per-breed accuracy."""
    per_class = {}
    for cls_idx, name in enumerate(class_names):
        mask = [i for i, t in enumerate(all_true) if t == cls_idx]
        if len(mask) == 0:
            per_class[name] = {"accuracy": 0.0, "total": 0, "correct": 0}
            continue
        correct = sum(1 for i in mask if all_preds[i] == cls_idx)
        per_class[name] = {
            "accuracy": correct / len(mask) * 100,
            "total": len(mask),
            "correct": correct,
        }
    return per_class


def plot_per_class_accuracy(per_class, top_n=30):
    """Plot top/bottom breed accuracies."""
    sorted_breeds = sorted(per_class.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    # Top N
    top = sorted_breeds[:top_n]
    names = [b[0] for b in top]
    accs = [b[1]["accuracy"] for b in top]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.35)))
    colors = ["green" if a >= 50 else "orange" if a >= 25 else "red" for a in accs]
    ax.barh(range(len(names)), accs, color=colors, alpha=0.7)
    ax.set(yticks=range(len(names)), yticklabels=names, 
           xlabel="Accuracy (%)", title=f"Top {top_n} Breed Accuracies")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(0, 105)
    
    for i, (acc, n) in enumerate(zip(accs, names)):
        total = per_class[n]["total"]
        ax.text(acc + 1, i, f"{acc:.1f}% ({total})", va="center", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "breed_classifier_per_class.png"), dpi=150)
    plt.close()
    print(f"  Saved: outputs/breed_classifier_per_class.png")


def save_metrics(class_names, class_counts, per_class, best_val_acc, best_epoch, history):
    """Save all metrics to JSON."""
    metrics = {
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_counts": class_counts,
        "per_class_accuracy": per_class,
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "history": history,
    }
    
    metrics_path = OUTPUT_DIR / "breed_classifier_metrics.json"
    with open(str(metrics_path), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: outputs/breed_classifier_metrics.json")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train Dog Breed Audio Classifier")
    parser.add_argument("--max-per-breed", type=int, default=700,
                        help="Max audio clips per breed (default: 700)")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Skip breeds with fewer clips (default: 10)")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Training epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"\n{'#' * 60}")
    print(f"  DOG BREED AUDIO CLASSIFIER - TRAINING")
    print(f"{'#' * 60}")
    
    # Step 1: Load data
    train_ds, val_ds, class_names, class_counts = load_breed_data(
        DATA_ROOT,
        max_per_breed=args.max_per_breed,
        min_samples=args.min_samples,
        val_split=args.val_split,
    )
    
    if len(class_names) < 2:
        print("ERROR: Need at least 2 breeds to train. Check data/breed_audio/")
        sys.exit(1)
    
    # Step 2: Train
    model, history, best_val_acc, best_epoch, all_preds, all_true = train_model(
        train_ds, val_ds,
        num_classes=len(class_names),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    
    # Step 3: Evaluate
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION")
    print(f"{'=' * 60}")
    
    per_class = compute_per_class_accuracy(all_true, all_preds, class_names)
    
    # Show top 10 and bottom 10
    sorted_breeds = sorted(per_class.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    print(f"\n  Top 10 breeds:")
    for name, info in sorted_breeds[:10]:
        print(f"    {name:35s} -> {info['accuracy']:6.2f}% ({info['correct']}/{info['total']})")
    
    print(f"\n  Bottom 10 breeds:")
    for name, info in sorted_breeds[-10:]:
        print(f"    {name:35s} -> {info['accuracy']:6.2f}% ({info['correct']}/{info['total']})")
    
    # Step 4: Save everything
    print(f"\n{'=' * 60}")
    print(f"  SAVING RESULTS")
    print(f"{'=' * 60}")
    
    plot_training_curves(history)
    plot_per_class_accuracy(per_class, top_n=min(30, len(class_names)))
    save_metrics(class_names, class_counts, per_class, best_val_acc, best_epoch, history)
    
    # Save class names separately for inference
    class_names_path = OUTPUT_DIR / "breed_class_names.json"
    with open(str(class_names_path), "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"  Saved: outputs/breed_class_names.json")
    
    print(f"\n{'#' * 60}")
    print(f"  DONE!")
    print(f"  Best accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"  Model saved: outputs/breed_classifier_best.pth")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
