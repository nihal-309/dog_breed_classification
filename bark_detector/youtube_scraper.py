"""
YouTube Bark Audio Scraper
============================
Downloads dog bark audio from YouTube with multi-layer filtering:

Layer 1: Search query optimization (breed-specific bark keywords)
Layer 2: Acoustic heuristic pre-filter (fast rejection of non-bark audio)
Layer 3: CNN bark detector (ML-based classification)
Layer 4: Optional manual validation queue

Pipeline:
  1. Search YouTube for "[breed] dog barking" videos
  2. Download audio with yt-dlp
  3. Segment into 4-second clips
  4. Filter each clip through acoustic heuristics + CNN bark detector
  5. Save passing clips to breed_audio/<breed>/

Usage:
    python -m bark_detector.youtube_scraper --breed labrador --max-videos 20
    python bark_detector/youtube_scraper.py --breed beagle --max-videos 10
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
import time
import random
import numpy as np
from pathlib import Path
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from bark_detector.model import (
    BarkDetector, load_wav, SAMPLE_RATE, DURATION, DEVICE
)
from bark_detector.acoustic_filter import is_likely_bark, compute_bark_score


# ============================================================================
# CONFIG
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
BREED_AUDIO_DIR = DATA_ROOT / "breed_audio"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Bark detector model path (must train first)
BARK_MODEL_PATH = OUTPUT_DIR / "bark_detector_best.pth"

# Audio settings
CLIP_DURATION = 4       # seconds per clip
CLIP_OVERLAP = 1        # seconds overlap between clips
MIN_CLIP_ENERGY = 0.005 # RMS threshold to skip silent clips

# Filtering thresholds
ACOUSTIC_THRESHOLD = 0.4    # heuristic filter (lower = more permissive)
CNN_THRESHOLD = 0.6          # CNN confidence threshold
COMBINED_THRESHOLD = 0.55    # combined score threshold

# Search query templates per breed
SEARCH_TEMPLATES = [
    "{breed} dog barking",
    "{breed} dog bark sound",
    "{breed} barking compilation",
    "{breed} puppy barking",
    "{breed} dog bark close up",
    "{breed} dog howling barking",
]


# ============================================================================
# YOUTUBE SEARCH & DOWNLOAD
# ============================================================================
def check_ytdlp():
    """Check if yt-dlp is available."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def search_youtube(query, max_results=10):
    """
    Search YouTube for videos matching the query.
    Returns list of video URLs.
    """
    try:
        cmd = [
            "yt-dlp",
            f"ytsearch{max_results}:{query}",
            "--get-id",
            "--no-warnings",
            "--flat-playlist",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"    WARNING: Search failed for '{query}'")
            return []

        video_ids = [vid.strip() for vid in result.stdout.strip().split("\n") if vid.strip()]
        urls = [f"https://www.youtube.com/watch?v={vid}" for vid in video_ids]
        return urls

    except subprocess.TimeoutExpired:
        print(f"    WARNING: Search timed out for '{query}'")
        return []
    except Exception as e:
        print(f"    WARNING: Search error for '{query}': {e}")
        return []


def download_audio(url, output_path, max_duration=120):
    """
    Download audio from a YouTube video as WAV.
    
    Args:
        url: YouTube video URL
        output_path: where to save the WAV file
        max_duration: max seconds to download (avoid very long videos)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            "yt-dlp",
            url,
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--output", str(output_path),
            "--no-playlist",
            "--no-warnings",
            "--quiet",
            # Limit duration to avoid downloading hour-long compilations
            "--match-filter", f"duration<={max_duration}",
            # Prefer lower quality (faster download, we only need audio)
            "--format", "worstaudio/worst",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        # yt-dlp may append .wav to the output path
        actual_path = Path(str(output_path))
        if not actual_path.exists():
            # Try with .wav extension added
            wav_path = Path(str(output_path) + ".wav")
            if wav_path.exists():
                wav_path.rename(actual_path)

        return actual_path.exists()

    except subprocess.TimeoutExpired:
        print(f"    Download timed out: {url}")
        return False
    except Exception as e:
        print(f"    Download error: {e}")
        return False


def get_video_info(url):
    """Get video metadata (title, duration) without downloading."""
    try:
        cmd = [
            "yt-dlp",
            url,
            "--dump-json",
            "--no-download",
            "--no-warnings",
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            info = json.loads(result.stdout.strip())
            return {
                "title": info.get("title", ""),
                "duration": info.get("duration", 0),
                "url": url,
            }
    except Exception:
        pass
    return None


# ============================================================================
# AUDIO SEGMENTATION
# ============================================================================
def segment_audio(wav_path, clip_duration=CLIP_DURATION, overlap=CLIP_OVERLAP,
                  sr=SAMPLE_RATE):
    """
    Split a WAV file into fixed-length clips.
    
    Args:
        wav_path: path to WAV file
        clip_duration: length of each clip in seconds
        overlap: overlap between clips in seconds
        sr: target sample rate
        
    Returns:
        list of numpy arrays (float32, mono, at sr)
    """
    try:
        orig_sr, data = wavfile.read(str(wav_path))
    except Exception as e:
        print(f"    Failed to read {wav_path}: {e}")
        return []

    # Convert to float32
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

    # Segment
    clip_samples = int(clip_duration * sr)
    hop_samples = int((clip_duration - overlap) * sr)
    clips = []

    pos = 0
    while pos + clip_samples <= len(data):
        clip = data[pos:pos + clip_samples]

        # Skip near-silent clips
        rms = np.sqrt(np.mean(clip ** 2))
        if rms >= MIN_CLIP_ENERGY:
            clips.append(clip)

        pos += hop_samples

    return clips


# ============================================================================
# MULTI-LAYER FILTERING
# ============================================================================
def filter_clips(clips, bark_detector=None, acoustic_threshold=ACOUSTIC_THRESHOLD,
                 cnn_threshold=CNN_THRESHOLD, combined_threshold=COMBINED_THRESHOLD):
    """
    Apply multi-layer filtering to audio clips.
    
    Layer 1: Acoustic heuristic filter (fast)
    Layer 2: CNN bark detector (if model available)
    Layer 3: Combined score
    
    Args:
        clips: list of numpy waveform arrays
        bark_detector: BarkDetector instance (or None to skip CNN)
        acoustic_threshold: minimum acoustic score
        cnn_threshold: minimum CNN confidence
        combined_threshold: minimum combined score
        
    Returns:
        list of (clip, combined_score) tuples that passed all filters
    """
    passed = []
    stats = {"total": len(clips), "acoustic_pass": 0, "cnn_pass": 0, "final_pass": 0}

    for clip in clips:
        # Layer 1: Acoustic heuristic
        result = compute_bark_score(clip)
        acoustic_score = result["score"]

        if acoustic_score < acoustic_threshold:
            continue
        stats["acoustic_pass"] += 1

        # Layer 2: CNN bark detector
        if bark_detector is not None:
            is_bark, cnn_confidence = bark_detector.predict_waveform(clip)
            if cnn_confidence < cnn_threshold:
                continue
            stats["cnn_pass"] += 1

            # Combined score (weighted: 40% acoustic, 60% CNN)
            combined = 0.4 * acoustic_score + 0.6 * cnn_confidence
        else:
            # No CNN available, use acoustic only with stricter threshold
            combined = acoustic_score
            if combined < 0.6:  # stricter when no CNN
                continue
            stats["cnn_pass"] += 1

        if combined >= combined_threshold:
            passed.append((clip, combined))
            stats["final_pass"] += 1

    return passed, stats


# ============================================================================
# SAVE CLIPS
# ============================================================================
def save_clip(clip, output_dir, breed, clip_index, score, sr=SAMPLE_RATE):
    """
    Save a filtered bark clip as WAV.
    
    Naming convention: yt_{breed}_{index}_{score}.wav
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to int16 for WAV
    clip_int16 = (clip * 32767).astype(np.int16)
    
    filename = f"yt_{breed}_{clip_index:05d}_s{int(score * 100):02d}.wav"
    filepath = output_dir / filename
    wavfile.write(str(filepath), sr, clip_int16)
    
    return filepath


# ============================================================================
# SCRAPE ONE BREED
# ============================================================================
def scrape_breed(breed, max_videos=20, bark_detector=None,
                 output_dir=None, target_clips=200):
    """
    Full pipeline for scraping bark audio for one breed.
    
    Args:
        breed: breed name (e.g., "labrador", "beagle")
        max_videos: max YouTube videos to download
        bark_detector: BarkDetector instance (or None)
        output_dir: where to save clips (default: breed_audio/<breed>/)
        target_clips: stop after collecting this many clips
        
    Returns:
        dict with scraping statistics
    """
    if output_dir is None:
        output_dir = BREED_AUDIO_DIR / breed.lower().replace(" ", "_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check existing clips
    existing = len(list(output_dir.glob("yt_*.wav")))
    print(f"\n  Existing YouTube clips for {breed}: {existing}")

    stats = {
        "breed": breed,
        "videos_searched": 0,
        "videos_downloaded": 0,
        "total_clips": 0,
        "clips_passed": 0,
        "clips_saved": existing,
        "errors": 0,
    }

    clip_counter = existing
    all_urls = []

    # Step 1: Search YouTube
    print(f"\n  Step 1: Searching YouTube for '{breed}' bark videos...")
    search_breed = breed.replace("_", " ")
    for template in SEARCH_TEMPLATES:
        query = template.format(breed=search_breed)
        urls = search_youtube(query, max_results=max_videos // len(SEARCH_TEMPLATES) + 1)
        all_urls.extend(urls)
        stats["videos_searched"] += len(urls)
        time.sleep(1)  # Rate limiting

    # Deduplicate
    all_urls = list(dict.fromkeys(all_urls))
    print(f"    Found {len(all_urls)} unique videos")

    if not all_urls:
        print(f"    No videos found for {breed}!")
        return stats

    # Step 2: Download and process each video
    print(f"\n  Step 2: Downloading and filtering...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for vid_idx, url in enumerate(all_urls):
            if stats["clips_saved"] >= target_clips + existing:
                print(f"\n    Reached target of {target_clips} new clips!")
                break

            print(f"\n    [{vid_idx + 1}/{len(all_urls)}] {url}")

            # Get video info
            info = get_video_info(url)
            if info:
                duration = info.get("duration", 0)
                title = info.get("title", "unknown")
                print(f"      Title: {title[:60]}...")
                print(f"      Duration: {duration}s")
                
                # Skip very short or very long videos
                if duration < 5:
                    print(f"      SKIP: Too short")
                    continue
                if duration > 300:
                    print(f"      SKIP: Too long (>5 min)")
                    continue

            # Download
            wav_path = Path(tmpdir) / f"video_{vid_idx}.wav"
            success = download_audio(url, wav_path)
            if not success:
                stats["errors"] += 1
                print(f"      FAILED to download")
                continue
            stats["videos_downloaded"] += 1

            # Segment
            clips = segment_audio(wav_path)
            stats["total_clips"] += len(clips)
            print(f"      Segmented into {len(clips)} clips")

            if not clips:
                continue

            # Filter
            passed_clips, filter_stats = filter_clips(
                clips, bark_detector,
                acoustic_threshold=ACOUSTIC_THRESHOLD,
                cnn_threshold=CNN_THRESHOLD,
                combined_threshold=COMBINED_THRESHOLD,
            )
            stats["clips_passed"] += len(passed_clips)
            print(f"      Filtering: {filter_stats['total']} -> "
                  f"acoustic:{filter_stats['acoustic_pass']} -> "
                  f"cnn:{filter_stats['cnn_pass']} -> "
                  f"final:{filter_stats['final_pass']}")

            # Save
            for clip, score in passed_clips:
                clip_counter += 1
                save_clip(clip, output_dir, breed.lower().replace(" ", "_"),
                          clip_counter, score)
                stats["clips_saved"] += 1

            # Clean up temp file
            if wav_path.exists():
                wav_path.unlink()

            # Rate limiting
            time.sleep(2)

    print(f"\n  Scraping complete for {breed}:")
    print(f"    Videos downloaded: {stats['videos_downloaded']}")
    print(f"    Total clips extracted: {stats['total_clips']}")
    print(f"    Clips passed filters: {stats['clips_passed']}")
    print(f"    Total clips saved: {stats['clips_saved']}")

    return stats


# ============================================================================
# BATCH SCRAPE MULTIPLE BREEDS
# ============================================================================
# Top breeds needing audio (from earlier analysis)
TARGET_BREEDS = [
    "labrador_retriever", "golden_retriever", "bulldog",
    "beagle", "poodle", "rottweiler",
    "yorkshire_terrier", "boxer", "dachshund",
    "pembroke_welsh_corgi", "doberman", "great_dane",
    "miniature_schnauzer", "australian_shepherd", "cavalier_king_charles",
    "shih_tzu", "boston_terrier", "pomeranian",
    "havanese", "english_springer_spaniel", "shetland_sheepdog",
    "bernese_mountain_dog", "border_collie", "cocker_spaniel",
    "vizsla",
]


def batch_scrape(breeds=None, max_videos_per_breed=20, target_clips_per_breed=200):
    """
    Scrape bark audio for multiple breeds.
    
    Args:
        breeds: list of breed names (default: TARGET_BREEDS)
        max_videos_per_breed: max YouTube videos per breed
        target_clips_per_breed: target clips per breed
    """
    if breeds is None:
        breeds = TARGET_BREEDS

    # Initialize bark detector if model exists
    bark_detector = None
    if BARK_MODEL_PATH.exists():
        print(f"  Loading bark detector from {BARK_MODEL_PATH}...")
        bark_detector = BarkDetector(BARK_MODEL_PATH, threshold=CNN_THRESHOLD)
        print(f"  Bark detector loaded!")
    else:
        print(f"  WARNING: No bark detector model found at {BARK_MODEL_PATH}")
        print(f"  Running with acoustic heuristics only (less accurate)")
        print(f"  Train the bark detector first: python -m bark_detector.train_bark_detector")

    all_stats = []

    for i, breed in enumerate(breeds):
        print(f"\n{'=' * 70}")
        print(f"  [{i + 1}/{len(breeds)}] Scraping: {breed}")
        print(f"{'=' * 70}")

        stats = scrape_breed(
            breed,
            max_videos=max_videos_per_breed,
            bark_detector=bark_detector,
            target_clips=target_clips_per_breed,
        )
        all_stats.append(stats)

        # Save progress
        with open(OUTPUT_DIR / "scraping_progress.json", "w") as f:
            json.dump(all_stats, f, indent=2)

        # Be nice to YouTube
        time.sleep(5)

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  BATCH SCRAPING COMPLETE")
    print(f"{'=' * 70}")
    
    total_saved = sum(s["clips_saved"] for s in all_stats)
    total_downloaded = sum(s["videos_downloaded"] for s in all_stats)
    total_errors = sum(s["errors"] for s in all_stats)
    
    print(f"  Breeds processed: {len(all_stats)}")
    print(f"  Videos downloaded: {total_downloaded}")
    print(f"  Total clips saved: {total_saved}")
    print(f"  Errors: {total_errors}")
    print(f"  Progress saved to: outputs/scraping_progress.json")

    return all_stats


# ============================================================================
# CLI
# ============================================================================
def main():
    global ACOUSTIC_THRESHOLD, CNN_THRESHOLD
    
    parser = argparse.ArgumentParser(description="YouTube Bark Audio Scraper")
    parser.add_argument("--breed", type=str, default=None,
                        help="Single breed to scrape (e.g., 'beagle', 'labrador_retriever')")
    parser.add_argument("--all", action="store_true",
                        help="Scrape all target breeds")
    parser.add_argument("--max-videos", type=int, default=20,
                        help="Max YouTube videos per breed (default: 20)")
    parser.add_argument("--target-clips", type=int, default=200,
                        help="Target clips per breed (default: 200)")
    parser.add_argument("--list-breeds", action="store_true",
                        help="List all target breeds")
    parser.add_argument("--acoustic-threshold", type=float, default=ACOUSTIC_THRESHOLD,
                        help=f"Acoustic filter threshold (default: {ACOUSTIC_THRESHOLD})")
    parser.add_argument("--cnn-threshold", type=float, default=CNN_THRESHOLD,
                        help=f"CNN confidence threshold (default: {CNN_THRESHOLD})")

    args = parser.parse_args()

    if args.list_breeds:
        print("\nTarget breeds for audio scraping:")
        for i, breed in enumerate(TARGET_BREEDS, 1):
            existing = BREED_AUDIO_DIR / breed
            count = len(list(existing.glob("*.wav"))) if existing.exists() else 0
            status = "has data" if count > 0 else "NEEDS DATA"
            print(f"  {i:2d}. {breed:<30s} [{count:5d} files] {status}")
        return

    # Check yt-dlp
    if not check_ytdlp():
        print("ERROR: yt-dlp is not installed!")
        print("Install it with: pip install yt-dlp")
        sys.exit(1)

    # Update thresholds from args
    ACOUSTIC_THRESHOLD = args.acoustic_threshold
    CNN_THRESHOLD = args.cnn_threshold

    if args.breed:
        # Single breed
        bark_detector = None
        if BARK_MODEL_PATH.exists():
            print(f"  Loading bark detector...")
            bark_detector = BarkDetector(BARK_MODEL_PATH, threshold=CNN_THRESHOLD)

        scrape_breed(
            args.breed,
            max_videos=args.max_videos,
            bark_detector=bark_detector,
            target_clips=args.target_clips,
        )
    elif args.all:
        # All breeds
        batch_scrape(
            max_videos_per_breed=args.max_videos,
            target_clips_per_breed=args.target_clips,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m bark_detector.youtube_scraper --breed beagle --max-videos 10")
        print("  python -m bark_detector.youtube_scraper --all --max-videos 20")
        print("  python -m bark_detector.youtube_scraper --list-breeds")


if __name__ == "__main__":
    main()
