"""
Audio Species Classifier - Cat vs Dog (Optimized)
===================================================
Pre-computes mel spectrograms into memory, then trains CNN.
This avoids recomputing spectrograms every epoch -> 10-20x faster.

Dataset: data/audio_organized
  train/cats/ (755)  |  train/dogs/ (658)
  val/cats/   (189)  |  val/dogs/   (165)
"""

import os
import sys
import time
import random
import warnings
import json
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

# ============================================================================
# CONFIG
# ============================================================================
DATA_ROOT    = Path(__file__).parent / "data" / "audio_organized"
OUTPUT_DIR   = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE  = 16000
DURATION     = 4
N_MELS       = 64
N_FFT        = 1024
HOP_LENGTH   = 512
BATCH_SIZE   = 32
EPOCHS       = 25
LR           = 1e-3
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES  = ["cats", "dogs"]
NUM_CLASSES  = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================================
# AUDIO PROCESSING
# ============================================================================
def load_wav(filepath):
    """Load wav, convert to mono float32, resample to 16kHz, pad/trim."""
    try:
        sr, data = wavfile.read(filepath)
    except Exception:
        return np.zeros(SAMPLE_RATE * DURATION, dtype=np.float32)

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    if data.ndim == 2:
        data = data.mean(axis=1)

    if sr != SAMPLE_RATE:
        num_samples = int(len(data) * SAMPLE_RATE / sr)
        data = np.interp(np.linspace(0, len(data) - 1, num_samples),
                         np.arange(len(data)), data).astype(np.float32)

    target_len = SAMPLE_RATE * DURATION
    if len(data) < target_len:
        data = np.pad(data, (0, target_len - len(data)), mode="constant")
    else:
        start = (len(data) - target_len) // 2
        data = data[start:start + target_len]

    return data


# Pre-compute mel filterbank once
def _build_mel_fb():
    fmax = SAMPLE_RATE / 2.0
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_pts = np.linspace(0, mel_max, N_MELS + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bins = np.floor((N_FFT + 1) * hz_pts / SAMPLE_RATE).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for i in range(N_MELS):
        l, c, r = bins[i], bins[i+1], bins[i+2]
        for j in range(l, c):
            if c != l: fb[i, j] = (j - l) / (c - l)
        for j in range(c, r):
            if r != c: fb[i, j] = (r - j) / (r - c)
    return fb

MEL_FB = _build_mel_fb()
WINDOW = np.hanning(N_FFT).astype(np.float32)


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
# PRE-COMPUTE ALL SPECTROGRAMS INTO MEMORY
# ============================================================================
def precompute_dataset(split):
    """Load all wavs, compute mel spectrograms, return as tensors."""
    specs, labels = [], []
    skipped = 0

    for label_idx, cls in enumerate(CLASS_NAMES):
        class_dir = DATA_ROOT / split / cls
        wav_files = sorted([f for f in class_dir.iterdir() if f.suffix == ".wav"])
        for i, fpath in enumerate(wav_files):
            wav = load_wav(fpath)
            mel = wav_to_mel(wav)
            if np.isnan(mel).any() or np.isinf(mel).any():
                skipped += 1
                continue
            specs.append(mel)
            labels.append(label_idx)
            if (i + 1) % 200 == 0:
                print(f"    {cls}: {i+1}/{len(wav_files)}")

    specs_t = torch.FloatTensor(np.stack(specs)).unsqueeze(1)  # (N,1,n_mels,T)
    labels_t = torch.LongTensor(labels)
    n_c = (labels_t == 0).sum().item()
    n_d = (labels_t == 1).sum().item()
    print(f"  [{split}] {len(labels)} spectrograms ({n_c} cats, {n_d} dogs) | shape: {tuple(specs_t.shape)} | skipped: {skipped}")
    return specs_t, labels_t


# ============================================================================
# DATASET WITH SPECAUGMENT
# ============================================================================
class AugmentedDataset(Dataset):
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
                spec[:, :, t0:t0+t_mask] = 0
            # Frequency masking
            if random.random() < 0.5:
                f = spec.shape[1]
                f_mask = random.randint(1, max(1, f // 8))
                f0 = random.randint(0, f - f_mask)
                spec[:, f0:f0+f_mask, :] = 0
            # Random gain
            if random.random() < 0.3:
                spec = spec + random.uniform(-0.3, 0.3)
        return spec, label


# ============================================================================
# MODEL
# ============================================================================
class AudioCNN(nn.Module):
    """4-block CNN: (B,1,64,T) -> (B,2). ~422K params."""
    def __init__(self):
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
            nn.Dropout(0.3), nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.gap(self.features(x)).flatten(1)
        return self.head(x)


# ============================================================================
# TRAINING
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
    return loss_sum/total, 100.0*correct/total, np.array(preds_all), np.array(targets_all)


# ============================================================================
# PLOTS
# ============================================================================
def save_plots(tl, vl, ta, va, preds, targets, output_dir):
    # Training curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(tl)+1)
    a1.plot(ep, tl, "b-o", ms=3, label="Train"); a1.plot(ep, vl, "r-o", ms=3, label="Val")
    a1.set(xlabel="Epoch", ylabel="Loss", title="Loss"); a1.legend(); a1.grid(True, alpha=0.3)
    a2.plot(ep, ta, "b-o", ms=3, label="Train"); a2.plot(ep, va, "r-o", ms=3, label="Val")
    a2.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy"); a2.legend(); a2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_dir/"training_curves.png", dpi=150); plt.close()

    # Confusion matrix
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True", title="Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=18)
    plt.tight_layout(); plt.savefig(output_dir/"confusion_matrix.png", dpi=150); plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("  AUDIO SPECIES CLASSIFIER - Cat vs Dog (Optimized)")
    print("  Pre-cached Mel Spectrograms + CNN | PyTorch")
    print("=" * 70)
    print(f"\n  Device: {DEVICE} | SR: {SAMPLE_RATE}Hz | Clip: {DURATION}s")
    print(f"  Mels: {N_MELS} | FFT: {N_FFT} | Hop: {HOP_LENGTH}")
    print(f"  Batch: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LR}")

    # ---- Phase 1: Pre-compute ----
    print(f"\n{'─'*50}")
    print("  Phase 1: Pre-computing spectrograms...")
    print(f"{'─'*50}")
    t0 = time.time()
    train_specs, train_labels = precompute_dataset("train")
    val_specs, val_labels = precompute_dataset("val")
    print(f"  Done in {time.time()-t0:.1f}s\n")

    # Save sample spectrograms
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    idxs = random.sample(range(len(train_labels)), 8)
    for i, idx in enumerate(idxs):
        ax = axes[i//4, i%4]
        ax.imshow(train_specs[idx, 0].numpy(), aspect="auto", origin="lower", cmap="magma")
        ax.set_title(CLASS_NAMES[train_labels[idx]], fontsize=10)
    plt.suptitle("Sample Mel Spectrograms", fontweight="bold")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/"sample_spectrograms.png", dpi=150); plt.close()
    print("  Saved sample_spectrograms.png")

    # Normalize
    mean, std = train_specs.mean(), train_specs.std()
    train_specs = (train_specs - mean) / (std + 1e-8)
    val_specs = (val_specs - mean) / (std + 1e-8)
    print(f"  Normalized: mean={mean:.4f}, std={std:.4f}")

    # DataLoaders
    train_loader = DataLoader(AugmentedDataset(train_specs, train_labels, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AugmentedDataset(val_specs, val_labels, augment=False),
                            batch_size=BATCH_SIZE, shuffle=False)

    # Class weights
    nc = (train_labels == 0).sum().item()
    nd = (train_labels == 1).sum().item()
    w = torch.FloatTensor([len(train_labels)/(2*nc), len(train_labels)/(2*nd)]).to(DEVICE)
    print(f"  Weights: cats={w[0]:.3f}, dogs={w[1]:.3f}")

    # ---- Phase 2: Model ----
    print(f"\n{'─'*50}")
    print("  Phase 2: Model")
    print(f"{'─'*50}")
    model = AudioCNN().to(DEVICE)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  AudioCNN: {nparams:,} params")

    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # ---- Phase 3: Train ----
    print(f"\n{'='*70}")
    print("  Phase 3: TRAINING")
    print(f"{'='*70}")

    best_acc, best_ep = 0.0, 0
    tl_hist, vl_hist, ta_hist, va_hist = [], [], [], []
    patience_ctr = 0
    PATIENCE = 8

    t_start = time.time()
    for ep in range(1, EPOCHS + 1):
        t_ep = time.time()
        tr_l, tr_a = train_epoch(model, train_loader, criterion, optimizer)
        vl_l, vl_a, vl_p, vl_t = evaluate(model, val_loader, criterion)
        scheduler.step(vl_l)

        tl_hist.append(tr_l); vl_hist.append(vl_l)
        ta_hist.append(tr_a); va_hist.append(vl_a)
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t_ep

        flag = ""
        if vl_a > best_acc:
            best_acc, best_ep = vl_a, ep
            patience_ctr = 0
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                         "val_acc": vl_a, "val_loss": vl_l,
                         "class_names": CLASS_NAMES,
                         "norm_mean": mean.item(), "norm_std": std.item()},
                        OUTPUT_DIR / "best_audio_model.pth")
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
    print(f"\n  Done in {total_t:.1f}s ({total_t/60:.1f} min)")
    print(f"  Best: {best_acc:.1f}% @ epoch {best_ep}")

    # ---- Phase 4: Evaluate ----
    print(f"\n{'='*70}")
    print("  Phase 4: EVALUATION")
    print(f"{'='*70}")

    ckpt = torch.load(OUTPUT_DIR/"best_audio_model.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    _, acc, preds, targets = evaluate(model, val_loader, criterion)

    print(f"\n  Val Accuracy: {acc:.2f}%\n")
    report = classification_report(targets, preds, target_names=CLASS_NAMES, digits=4)
    for line in report.split("\n"):
        print(f"  {line}")

    # ---- Phase 5: Save ----
    save_plots(tl_hist, vl_hist, ta_hist, va_hist, preds, targets, OUTPUT_DIR)
    print(f"\n  Saved training_curves.png")
    print(f"  Saved confusion_matrix.png")

    summary = {
        "model": "AudioCNN", "classes": CLASS_NAMES,
        "train_samples": len(train_labels), "val_samples": len(val_labels),
        "best_epoch": best_ep, "best_val_acc": round(best_acc, 2),
        "total_params": nparams, "training_time_sec": round(total_t, 1),
        "norm_mean": round(mean.item(), 6), "norm_std": round(std.item(), 6),
        "config": {"sample_rate": SAMPLE_RATE, "duration": DURATION,
                   "n_mels": N_MELS, "n_fft": N_FFT, "hop_length": HOP_LENGTH,
                   "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR}
    }
    with open(OUTPUT_DIR/"training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  ALL DONE!")
    print(f"  Model:   outputs/best_audio_model.pth")
    print(f"  Curves:  outputs/training_curves.png")
    print(f"  Matrix:  outputs/confusion_matrix.png")
    print(f"  Summary: outputs/training_summary.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
