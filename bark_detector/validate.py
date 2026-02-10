"""
Manual Validation Tool
========================
Layer 4 of the bark detection pipeline.
Provides CLI-based review of scraped audio clips for manual QA.

Features:
  - Audio playback (if sounddevice is available)
  - Visual waveform + spectrogram display
  - Accept/reject/skip interface
  - Batch statistics tracking
  - Moves rejected files to quarantine folder

Usage:
    python -m bark_detector.validate --breed beagle
    python -m bark_detector.validate --dir data/breed_audio/beagle --sample 50
"""

import os
import sys
import argparse
import json
import shutil
import numpy as np
from pathlib import Path
from scipy.io import wavfile

import matplotlib
matplotlib.use("TkAgg")  # Use interactive backend for display
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from bark_detector.model import load_wav, wav_to_mel, SAMPLE_RATE, DURATION
from bark_detector.acoustic_filter import compute_bark_score

# ============================================================================
# CONFIG
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
BREED_AUDIO_DIR = DATA_ROOT / "breed_audio"
QUARANTINE_DIR = DATA_ROOT / "quarantine"


# ============================================================================
# AUDIO PLAYBACK
# ============================================================================
def play_audio(filepath):
    """Try to play audio using available library."""
    try:
        import sounddevice as sd
        sr, data = wavfile.read(str(filepath))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        if data.ndim == 2:
            data = data.mean(axis=1)
        sd.play(data, sr)
        sd.wait()
        return True
    except ImportError:
        print("    (sounddevice not installed - run: pip install sounddevice)")
        return False
    except Exception as e:
        print(f"    Playback error: {e}")
        return False


# ============================================================================
# DISPLAY
# ============================================================================
def show_clip_info(filepath, index, total):
    """Display waveform, spectrogram, and acoustic features for a clip."""
    wav = load_wav(str(filepath))
    mel = wav_to_mel(wav)
    score_info = compute_bark_score(wav)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Waveform
    t = np.linspace(0, DURATION, len(wav))
    axes[0].plot(t, wav, linewidth=0.5, color="steelblue")
    axes[0].set(xlabel="Time (s)", ylabel="Amplitude", title="Waveform")
    axes[0].grid(True, alpha=0.3)

    # Spectrogram
    axes[1].imshow(mel, aspect="auto", origin="lower", cmap="magma")
    axes[1].set(xlabel="Time", ylabel="Mel Bin", title="Mel Spectrogram")

    # Feature scores
    features = score_info.get("features", {})
    feature_names = list(features.keys())
    feature_vals = [features[k] for k in feature_names]
    
    # Normalize for display
    if feature_vals:
        bars = axes[2].barh(range(len(feature_names)), feature_vals, color="teal", alpha=0.7)
        axes[2].set(yticks=range(len(feature_names)),
                    yticklabels=[n.replace("_", " ") for n in feature_names],
                    xlabel="Value", title=f"Bark Score: {score_info['score']:.3f}")
        axes[2].grid(True, alpha=0.3, axis="x")

    fig.suptitle(f"[{index}/{total}] {filepath.name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    return score_info


# ============================================================================
# VALIDATION LOOP
# ============================================================================
def validate_clips(audio_dir, sample_size=None, pattern="yt_*.wav"):
    """
    Interactive validation of audio clips.
    
    Controls:
      a/y  = Accept (keep the clip)
      r/n  = Reject (move to quarantine)
      p    = Play audio
      s    = Skip (leave undecided)
      q    = Quit
      
    Args:
        audio_dir: directory containing clips
        sample_size: validate random sample (None = all)
        pattern: glob pattern for files to validate
    """
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        print(f"Directory not found: {audio_dir}")
        return

    # Collect files
    files = sorted(audio_dir.glob(pattern))
    if not files:
        print(f"No files matching '{pattern}' in {audio_dir}")
        return

    if sample_size and sample_size < len(files):
        import random
        files = random.sample(files, sample_size)
        print(f"  Sampled {sample_size} files from {len(list(audio_dir.glob(pattern)))}")

    # Quarantine directory
    breed_name = audio_dir.name
    quarantine = QUARANTINE_DIR / breed_name
    quarantine.mkdir(parents=True, exist_ok=True)

    # Stats
    stats = {"total": len(files), "accepted": 0, "rejected": 0, "skipped": 0}
    log = []

    print(f"\n{'=' * 60}")
    print(f"  MANUAL VALIDATION: {breed_name}")
    print(f"  Files to review: {len(files)}")
    print(f"  Quarantine: {quarantine}")
    print(f"{'=' * 60}")
    print(f"  Controls: [a]ccept | [r]eject | [p]lay | [s]kip | [q]uit")
    print(f"{'=' * 60}\n")

    for idx, filepath in enumerate(files, 1):
        print(f"\n  --- [{idx}/{len(files)}] {filepath.name} ---")

        # Show info
        try:
            score_info = show_clip_info(filepath, idx, len(files))
            bark_score = score_info["score"]
            print(f"    Bark score: {bark_score:.3f}")
            for reason in score_info.get("reasons", []):
                print(f"      - {reason}")
        except Exception as e:
            print(f"    Display error: {e}")
            bark_score = -1

        # Input loop
        while True:
            action = input(f"    Action [a/r/p/s/q]: ").strip().lower()

            if action in ("a", "y", "accept", "yes"):
                stats["accepted"] += 1
                log.append({"file": filepath.name, "action": "accepted", "score": bark_score})
                print(f"    -> ACCEPTED")
                break
            elif action in ("r", "n", "reject", "no"):
                # Move to quarantine
                dest = quarantine / filepath.name
                shutil.move(str(filepath), str(dest))
                stats["rejected"] += 1
                log.append({"file": filepath.name, "action": "rejected", "score": bark_score})
                print(f"    -> REJECTED (moved to quarantine)")
                break
            elif action == "p":
                play_audio(filepath)
            elif action in ("s", "skip"):
                stats["skipped"] += 1
                log.append({"file": filepath.name, "action": "skipped", "score": bark_score})
                print(f"    -> SKIPPED")
                break
            elif action in ("q", "quit", "exit"):
                print(f"\n  Quitting early...")
                stats["skipped"] += len(files) - idx
                break
            else:
                print(f"    Unknown action. Use: a(ccept), r(eject), p(lay), s(kip), q(uit)")

        plt.close("all")

        if action in ("q", "quit", "exit"):
            break

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  VALIDATION COMPLETE: {breed_name}")
    print(f"{'=' * 60}")
    print(f"  Reviewed:  {stats['accepted'] + stats['rejected'] + stats['skipped']}/{stats['total']}")
    print(f"  Accepted:  {stats['accepted']}")
    print(f"  Rejected:  {stats['rejected']}")
    print(f"  Skipped:   {stats['skipped']}")
    if stats['accepted'] + stats['rejected'] > 0:
        accept_rate = stats['accepted'] / (stats['accepted'] + stats['rejected']) * 100
        print(f"  Accept Rate: {accept_rate:.1f}%")
    print(f"  Quarantine: {quarantine}")

    # Save log
    log_path = PROJECT_ROOT / "outputs" / f"validation_log_{breed_name}.json"
    with open(str(log_path), "w") as f:
        json.dump({"stats": stats, "log": log}, f, indent=2)
    print(f"  Log saved: {log_path}")

    return stats


# ============================================================================
# AUTO-VALIDATION (non-interactive)
# ============================================================================
def auto_validate(audio_dir, acoustic_threshold=0.45, pattern="yt_*.wav"):
    """
    Automatic validation using acoustic score only.
    Moves low-scoring clips to quarantine without manual review.
    
    Good for quick cleanup before manual spot-checking.
    """
    audio_dir = Path(audio_dir)
    files = sorted(audio_dir.glob(pattern))
    
    if not files:
        print(f"No files matching '{pattern}' in {audio_dir}")
        return

    breed_name = audio_dir.name
    quarantine = QUARANTINE_DIR / breed_name
    quarantine.mkdir(parents=True, exist_ok=True)

    accepted = 0
    rejected = 0

    print(f"\n  Auto-validating {len(files)} clips (threshold={acoustic_threshold})...")

    for filepath in files:
        wav = load_wav(str(filepath))
        result = compute_bark_score(wav)
        
        if result["score"] < acoustic_threshold:
            dest = quarantine / filepath.name
            shutil.move(str(filepath), str(dest))
            rejected += 1
        else:
            accepted += 1

    print(f"  Auto-validation: {accepted} accepted, {rejected} rejected")
    print(f"  Quarantine: {quarantine}")

    return {"accepted": accepted, "rejected": rejected}


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Manual Bark Audio Validation Tool")
    parser.add_argument("--breed", type=str, default=None,
                        help="Breed to validate (e.g., 'beagle')")
    parser.add_argument("--dir", type=str, default=None,
                        help="Directory to validate (overrides --breed)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random sample size (default: all)")
    parser.add_argument("--pattern", type=str, default="yt_*.wav",
                        help="Glob pattern for files (default: yt_*.wav)")
    parser.add_argument("--auto", action="store_true",
                        help="Run automatic validation (no manual review)")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Threshold for auto-validation (default: 0.45)")

    args = parser.parse_args()

    if args.dir:
        audio_dir = Path(args.dir)
    elif args.breed:
        audio_dir = BREED_AUDIO_DIR / args.breed
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m bark_detector.validate --breed beagle")
        print("  python -m bark_detector.validate --breed beagle --sample 20")
        print("  python -m bark_detector.validate --dir data/breed_audio/beagle --auto")
        return

    if args.auto:
        auto_validate(audio_dir, acoustic_threshold=args.threshold, pattern=args.pattern)
    else:
        validate_clips(audio_dir, sample_size=args.sample, pattern=args.pattern)


if __name__ == "__main__":
    main()
