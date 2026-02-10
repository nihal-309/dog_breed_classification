"""
Download Labrador audio files from Barkopedia-Dog-Vocal-Detection
Uses direct HuggingFace API to list and download raw wav files.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# CONFIG
# ============================================================================
REPO_ID = "ArlingtonCL2/Barkopedia-Dog-Vocal-Detection"
BASE_API = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main"
BASE_RAW = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

OUTPUT_DIR = Path(r"C:\Users\Nihal\Desktop\DAA\question_papers\dog_breed_classification\data\breed_audio\labrador")

# Folders in the repo that contain labrador files
LABRADOR_FOLDERS = [
    "unlabeled_audios/labrador",
    "weak_audios/labrador",
]

# strong_audios has breed in filename suffix (e.g. _labrador.wav)
STRONG_AUDIOS_FOLDER = "strong_audios"


def fetch_json(url, retries=3):
    """Fetch JSON from URL with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Python/3"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed to fetch {url}: {e}")
                return None


def list_files_in_folder(folder_path):
    """List all files in a HF repo folder via API."""
    url = f"{BASE_API}/{folder_path}"
    data = fetch_json(url)
    if not data:
        return []

    files = []
    for item in data:
        if item.get("type") == "file" and item["path"].endswith(".wav"):
            files.append(item["path"])
    return files


def download_file(file_path, dest_dir):
    """Download a single file from HuggingFace."""
    url = f"{BASE_RAW}/{urllib.request.quote(file_path, safe='/')}"
    filename = os.path.basename(file_path)
    dest = dest_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        return filename, "skipped"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Python/3"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            dest.write_bytes(data)
        return filename, "ok"
    except Exception as e:
        return filename, f"error: {e}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_files = []

    # 1) List files from labrador subfolders
    for folder in LABRADOR_FOLDERS:
        print(f"  Listing: {folder} ...")
        files = list_files_in_folder(folder)
        print(f"    Found {len(files)} wav files")
        all_files.extend(files)

    # 2) List strong_audios and filter for labrador
    print(f"  Listing: {STRONG_AUDIOS_FOLDER} ...")
    strong_files = list_files_in_folder(STRONG_AUDIOS_FOLDER)
    labrador_strong = [f for f in strong_files if "_labrador" in f.lower()]
    print(f"    Found {len(labrador_strong)} labrador wav files (of {len(strong_files)} total)")
    all_files.extend(labrador_strong)

    print(f"\n  Total labrador files to download: {len(all_files)}")

    if not all_files:
        print("  No files found! Exiting.")
        return

    # 3) Download with threading
    print(f"  Downloading to: {OUTPUT_DIR}\n")
    downloaded = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(download_file, fp, OUTPUT_DIR): fp
            for fp in all_files
        }

        for i, future in enumerate(as_completed(futures), 1):
            fname, status = future.result()
            if status == "ok":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"    ERROR: {fname}: {status}")

            if i % 20 == 0 or i == len(all_files):
                print(f"    Progress: {i}/{len(all_files)} "
                      f"(downloaded={downloaded}, skipped={skipped}, errors={errors})")

    # Final count
    total_wavs = len(list(OUTPUT_DIR.glob("*.wav")))
    print(f"\n  Done! {total_wavs} labrador wav files in {OUTPUT_DIR}")


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    print("\n  BARKOPEDIA LABRADOR DOWNLOADER")
    print("  " + "=" * 40 + "\n")
    main()
