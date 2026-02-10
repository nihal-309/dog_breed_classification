"""
Organize DogSpeak + Barkopedia into breed-specific folders
============================================================
1. Reorganize DogSpeak (dog_1..dog_156) → 5 breed folders
2. Download Labrador audio from Barkopedia-Dog-Vocal-Detection
3. Combine into 6 breed folders ready for training

Output: data/breed_audio/
    ├── chihuahua/
    ├── german_shepherd/
    ├── husky/
    ├── labrador/
    ├── pitbull/
    └── shiba_inu/
"""

import os
import sys
import shutil
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================
PROJECT_ROOT = Path(r"C:\Users\Nihal\Desktop\DAA\question_papers\dog_breed_classification")
DOGSPEAK_DIR = PROJECT_ROOT / "DogSpeak_Dataset" / "dogspeak_released"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "breed_audio"

# Map DogSpeak breed codes → clean folder names
BREED_MAP = {
    "chihuahua": "chihuahua",
    "gsd":       "german_shepherd",
    "husky":     "husky",
    "pitbull":   "pitbull",
    "shibainu":  "shiba_inu",
}


def extract_breed_from_filename(filename):
    """
    Extract breed from DogSpeak filename format:
    e.g. '0_chihuahua_M_dog_1.wav' → 'chihuahua'
         '38218_gsd_M_dog_50.wav'  → 'gsd'
    """
    parts = filename.replace(".wav", "").split("_")
    # Format: [idx, breed, sex, 'dog', id]
    # breed is always parts[1] (single token in this dataset)
    if len(parts) >= 4:
        return parts[1]
    return None


def step1_organize_dogspeak():
    """Reorganize DogSpeak from per-dog folders to per-breed folders."""
    print("=" * 60)
    print("  STEP 1: Organizing DogSpeak into breed folders")
    print("=" * 60)

    if not DOGSPEAK_DIR.exists():
        print(f"  ERROR: DogSpeak not found at {DOGSPEAK_DIR}")
        return False

    breed_counts = {v: 0 for v in BREED_MAP.values()}
    skipped = 0

    # Iterate all dog_X folders
    dog_dirs = sorted([d for d in DOGSPEAK_DIR.iterdir() if d.is_dir()])
    print(f"  Found {len(dog_dirs)} dog folders in DogSpeak")

    for dog_dir in dog_dirs:
        wav_files = list(dog_dir.glob("*.wav"))
        for wav_file in wav_files:
            breed_code = extract_breed_from_filename(wav_file.name)
            if breed_code and breed_code in BREED_MAP:
                breed_name = BREED_MAP[breed_code]
                dest_dir = OUTPUT_DIR / breed_name
                dest_dir.mkdir(parents=True, exist_ok=True)

                dest_file = dest_dir / wav_file.name
                if not dest_file.exists():
                    shutil.copy2(wav_file, dest_file)
                breed_counts[breed_name] += 1
            else:
                skipped += 1

    print(f"\n  DogSpeak organization complete:")
    for breed, count in sorted(breed_counts.items()):
        print(f"    {breed:20s}: {count:>6,} files")
    print(f"    {'SKIPPED':20s}: {skipped:>6} files")
    total = sum(breed_counts.values())
    print(f"    {'TOTAL':20s}: {total:>6,} files")
    return True


def step2_download_barkopedia_labrador():
    """Download Labrador audio from Barkopedia-Dog-Vocal-Detection."""
    print(f"\n{'=' * 60}")
    print("  STEP 2: Downloading Labrador from Barkopedia")
    print("=" * 60)

    labrador_dir = OUTPUT_DIR / "labrador"
    labrador_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(labrador_dir.glob("*.wav"))
    if len(existing) > 50:
        print(f"  Labrador folder already has {len(existing)} files, skipping download.")
        return True

    try:
        from datasets import load_dataset
    except ImportError:
        print("  Installing 'datasets' library...")
        os.system(f'"{sys.executable}" -m pip install datasets soundfile')
        from datasets import load_dataset

    print("  Loading Barkopedia-Dog-Vocal-Detection dataset...")
    print("  (This may take a few minutes on first download)\n")

    labrador_count = 0

    # The dataset has multiple splits/folders. Let's try loading and filtering.
    try:
        # Try loading the full dataset
        ds = load_dataset(
            "ArlingtonCL2/Barkopedia-Dog-Vocal-Detection"
        )
        print(f"  Dataset loaded. Splits: {list(ds.keys())}")

        # Look through all splits
        for split_name, split_data in ds.items():
            print(f"\n  Processing split: {split_name} ({len(split_data)} items)")

            for i, item in enumerate(split_data):
                # Check if this item is labrador
                # The audio files have breed in filename or path
                audio_path = ""
                if "audio" in item and isinstance(item["audio"], dict):
                    audio_path = item["audio"].get("path", "")
                elif "path" in item:
                    audio_path = item["path"]
                elif "file" in item:
                    audio_path = item["file"]

                is_labrador = "labrador" in str(audio_path).lower()

                # Also check any label/metadata columns
                for key in item:
                    if isinstance(item[key], str) and "labrador" in item[key].lower():
                        is_labrador = True
                        break

                if is_labrador:
                    # Save the audio
                    if "audio" in item and isinstance(item["audio"], dict):
                        audio_data = item["audio"]
                        array = audio_data.get("array")
                        sr = audio_data.get("sampling_rate", 16000)

                        if array is not None:
                            import numpy as np
                            import soundfile as sf

                            fname = f"labrador_bark_{labrador_count:05d}.wav"
                            if audio_path:
                                # Use original filename if available
                                base_name = os.path.basename(audio_path)
                                if base_name.endswith(".wav"):
                                    fname = base_name

                            out_path = labrador_dir / fname
                            if not out_path.exists():
                                sf.write(str(out_path), array, sr)
                            labrador_count += 1

                if (i + 1) % 100 == 0:
                    print(f"    Processed {i+1}/{len(split_data)}, found {labrador_count} labrador clips so far")

    except Exception as e:
        print(f"  Error with HuggingFace datasets API: {e}")
        print("  Trying alternative download method...")
        return step2_download_barkopedia_labrador_alt()

    print(f"\n  Labrador download complete: {labrador_count} audio files saved")
    return labrador_count > 0


def step2_download_barkopedia_labrador_alt():
    """Alternative: download via huggingface_hub if datasets API fails."""
    from huggingface_hub import hf_hub_download, list_repo_tree

    labrador_dir = OUTPUT_DIR / "labrador"
    labrador_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "ArlingtonCL2/Barkopedia-Dog-Vocal-Detection"
    print(f"  Listing files in {repo_id}...")

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)

        labrador_files = []
        for f in files:
            path = f.rfilename if hasattr(f, 'rfilename') else str(f)
            if "labrador" in path.lower() and path.endswith(".wav"):
                labrador_files.append(path)

        print(f"  Found {len(labrador_files)} labrador .wav files")

        for i, fpath in enumerate(labrador_files):
            try:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=fpath,
                    repo_type="dataset"
                )
                dest = labrador_dir / os.path.basename(fpath)
                if not dest.exists():
                    shutil.copy2(local, dest)
                if (i + 1) % 50 == 0:
                    print(f"    Downloaded {i+1}/{len(labrador_files)}")
            except Exception as e:
                print(f"    Error downloading {fpath}: {e}")

        final_count = len(list(labrador_dir.glob("*.wav")))
        print(f"\n  Labrador download complete: {final_count} files")
        return final_count > 0

    except Exception as e:
        print(f"  Alternative download also failed: {e}")
        print("  Please manually download labrador files from:")
        print(f"  https://huggingface.co/datasets/{repo_id}")
        return False


def step3_print_summary():
    """Print final summary of all breed folders."""
    print(f"\n{'=' * 60}")
    print("  FINAL SUMMARY: Breed Audio Dataset")
    print("=" * 60)
    print(f"  Location: {OUTPUT_DIR}\n")

    total = 0
    if OUTPUT_DIR.exists():
        for breed_dir in sorted(OUTPUT_DIR.iterdir()):
            if breed_dir.is_dir():
                count = len(list(breed_dir.glob("*.wav")))
                total += count
                bar = "█" * (count // 500) + "░" * max(0, 30 - count // 500)
                print(f"    {breed_dir.name:20s}: {count:>6,} files  {bar}")

    print(f"\n    {'TOTAL':20s}: {total:>6,} files")
    print(f"\n  Ready for breed-based audio classification training! 🎵🐕")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)

    print("\n" + "=" * 60)
    print("  BREED AUDIO DATASET ORGANIZER")
    print("  DogSpeak (5 breeds) + Barkopedia Labrador (6th breed)")
    print("=" * 60 + "\n")

    # Step 1: Organize DogSpeak into breed folders
    ok = step1_organize_dogspeak()
    if not ok:
        print("  Failed at step 1. Exiting.")
        sys.exit(1)

    # Step 2: Download Labrador from Barkopedia
    step2_download_barkopedia_labrador()

    # Step 3: Summary
    step3_print_summary()
