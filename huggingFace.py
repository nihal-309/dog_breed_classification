"""
DogSpeak Dataset - Resume Download Script
Downloads ONLY missing files by comparing local vs remote.
"""
import os
import sys
import time
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ──
REPO_ID = "ArlingtonCL2/DogSpeak_Dataset"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "DogSpeak_Dataset")
RELEASED_DIR = os.path.join(LOCAL_DIR, "dogspeak_released")
MAX_WORKERS = 4  # parallel downloads (keep low to avoid rate limits)

api = HfApi()


def get_remote_files_for_folder(folder_name):
    """Get list of all files in a remote dog folder."""
    path = f"dogspeak_released/{folder_name}"
    try:
        items = list(api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=path))
        return [item.path for item in items if hasattr(item, 'size')]  # files only
    except Exception as e:
        print(f"  ⚠ Error listing {folder_name}: {e}")
        return []


def get_local_files(folder_name):
    """Get set of filenames already downloaded for a dog folder."""
    folder_path = os.path.join(RELEASED_DIR, folder_name)
    if not os.path.exists(folder_path):
        return set()
    return set(os.listdir(folder_path))


def download_file(remote_path):
    """Download a single file from HuggingFace."""
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=remote_path,
            local_dir=LOCAL_DIR,
        )
        return True, remote_path
    except Exception as e:
        return False, f"{remote_path}: {e}"


def main():
    print("=" * 60)
    print("  DogSpeak Dataset - Resume Download")
    print("=" * 60)
    print(f"\nLocal dir: {RELEASED_DIR}")

    # ── Step 1: Download metadata.csv if missing ──
    meta_path = os.path.join(LOCAL_DIR, "metadata.csv")
    if not os.path.exists(meta_path):
        print("\n📥 Downloading metadata.csv...")
        try:
            hf_hub_download(REPO_ID, repo_type="dataset", filename="metadata.csv", local_dir=LOCAL_DIR)
            print("  ✅ metadata.csv downloaded")
        except Exception as e:
            print(f"  ⚠ Could not download metadata.csv: {e}")

    # ── Step 2: Get all remote folders ──
    print("\n🔍 Listing remote folders...")
    remote_items = list(api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo="dogspeak_released"))
    remote_folders = sorted([item.path.split("/")[-1] for item in remote_items])
    print(f"  Remote folders: {len(remote_folders)}")

    # ── Step 3: Compare each folder ──
    print("\n📊 Comparing local vs remote for each folder...\n")
    all_missing_files = []
    folders_complete = 0
    folders_partial = 0
    folders_empty = 0

    for i, folder in enumerate(remote_folders, 1):
        remote_files = get_remote_files_for_folder(folder)
        remote_filenames = {os.path.basename(f) for f in remote_files}
        remote_path_map = {os.path.basename(f): f for f in remote_files}

        local_files = get_local_files(folder)

        missing = remote_filenames - local_files
        total_remote = len(remote_filenames)
        total_local = len(local_files)

        if len(missing) == 0:
            status = "✅"
            folders_complete += 1
        elif total_local == 0:
            status = "❌ MISSING"
            folders_empty += 1
        else:
            status = f"⚠️  PARTIAL"
            folders_partial += 1

        if missing:
            print(f"  [{i:3d}/{len(remote_folders)}] {folder:10s}: {total_local:5d}/{total_remote:5d} files  {status} ({len(missing)} to download)")
            for fname in missing:
                all_missing_files.append(remote_path_map[fname])
        else:
            print(f"  [{i:3d}/{len(remote_folders)}] {folder:10s}: {total_local:5d}/{total_remote:5d} files  {status}")

    # ── Step 4: Summary ──
    print(f"\n{'=' * 60}")
    print(f"  Summary:")
    print(f"    Complete folders:  {folders_complete}")
    print(f"    Partial folders:   {folders_partial}")
    print(f"    Missing folders:   {folders_empty}")
    print(f"    Total files to download: {len(all_missing_files)}")
    print(f"{'=' * 60}")

    if not all_missing_files:
        print("\n🎉 Dataset is complete! Nothing to download.")
        return

    # ── Step 5: Download missing files ──
    answer = input(f"\nDownload {len(all_missing_files)} missing files? [Y/n]: ").strip().lower()
    if answer and answer != 'y':
        print("Aborted.")
        return

    print(f"\n📥 Downloading {len(all_missing_files)} files ({MAX_WORKERS} parallel workers)...\n")
    
    downloaded = 0
    failed = 0
    start_time = time.time()
    failed_files = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, f): f for f in all_missing_files}
        
        for future in as_completed(futures):
            success, result = future.result()
            if success:
                downloaded += 1
            else:
                failed += 1
                failed_files.append(result)
            
            total_done = downloaded + failed
            elapsed = time.time() - start_time
            rate = total_done / elapsed if elapsed > 0 else 0
            remaining = (len(all_missing_files) - total_done) / rate if rate > 0 else 0
            
            if total_done % 50 == 0 or total_done == len(all_missing_files):
                print(f"  Progress: {total_done}/{len(all_missing_files)} "
                      f"({downloaded} ok, {failed} failed) "
                      f"[{rate:.1f} files/s, ~{remaining/60:.1f}min remaining]")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Download Complete!")
    print(f"    Downloaded: {downloaded}")
    print(f"    Failed:     {failed}")
    print(f"    Time:       {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")

    if failed_files:
        print(f"\n⚠ Failed files:")
        for f in failed_files[:20]:
            print(f"  - {f}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more")


if __name__ == "__main__":
    main()