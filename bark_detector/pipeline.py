"""
Bark Detection Pipeline Orchestrator
=======================================
Ties all layers together into a single run command.

Full Pipeline:
  Step 1: Train bark detector on existing data (if no model exists)
  Step 2: Scrape YouTube for target breeds
  Step 3: Auto-validate scraped clips
  Step 4: Report results

Usage:
    python -m bark_detector.pipeline --step train
    python -m bark_detector.pipeline --step scrape --breed beagle
    python -m bark_detector.pipeline --step scrape-all
    python -m bark_detector.pipeline --step validate --breed beagle
    python -m bark_detector.pipeline --step full            # run everything
    python -m bark_detector.pipeline --step status          # check progress
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CONFIG
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
BREED_AUDIO_DIR = DATA_ROOT / "breed_audio"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
BARK_MODEL_PATH = OUTPUT_DIR / "bark_detector_best.pth"

# Target breeds (same list as youtube_scraper.py)
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

# Minimum clips needed per breed
MIN_CLIPS_PER_BREED = 150


# ============================================================================
# STATUS CHECK
# ============================================================================
def check_status():
    """Print overview of data collection progress."""
    print(f"\n{'=' * 70}")
    print(f"  BARK DETECTION PIPELINE - Status")
    print(f"{'=' * 70}")

    # Model status
    if BARK_MODEL_PATH.exists():
        summary_path = OUTPUT_DIR / "bark_detector_summary.json"
        if summary_path.exists():
            with open(str(summary_path)) as f:
                summary = json.load(f)
            print(f"\n  Bark Detector Model: TRAINED")
            print(f"    Accuracy: {summary.get('best_val_acc', '?')}%")
            print(f"    Epoch: {summary.get('best_epoch', '?')}")
            print(f"    Params: {summary.get('total_params', '?'):,}")
        else:
            print(f"\n  Bark Detector Model: EXISTS (no summary)")
    else:
        print(f"\n  Bark Detector Model: NOT TRAINED")
        print(f"    Run: python -m bark_detector.pipeline --step train")

    # Breed audio status
    print(f"\n  {'Breed':<30s} {'Clips':>7s} {'Status':<15s}")
    print(f"  {'─' * 55}")

    # Existing breeds (already have data)
    existing_breeds = []
    for d in sorted(BREED_AUDIO_DIR.iterdir()) if BREED_AUDIO_DIR.exists() else []:
        if d.is_dir():
            count = len(list(d.glob("*.wav")))
            existing_breeds.append((d.name, count))

    for name, count in existing_breeds:
        if name not in [b.lower().replace(" ", "_") for b in TARGET_BREEDS]:
            status = "READY" if count >= MIN_CLIPS_PER_BREED else f"NEED {MIN_CLIPS_PER_BREED - count} more"
            print(f"  {name:<30s} {count:>7d} {status:<15s}")

    # Target breeds
    total_needed = 0
    total_have = 0
    breeds_complete = 0
    breeds_incomplete = 0

    for breed in TARGET_BREEDS:
        breed_dir = BREED_AUDIO_DIR / breed
        count = len(list(breed_dir.glob("*.wav"))) if breed_dir.exists() else 0
        total_have += count
        
        if count >= MIN_CLIPS_PER_BREED:
            status = "READY"
            breeds_complete += 1
        elif count > 0:
            needed = MIN_CLIPS_PER_BREED - count
            total_needed += needed
            status = f"NEED {needed} more"
            breeds_incomplete += 1
        else:
            total_needed += MIN_CLIPS_PER_BREED
            status = "NO DATA"
            breeds_incomplete += 1

        print(f"  {breed:<30s} {count:>7d} {status:<15s}")

    print(f"\n  Summary:")
    print(f"    Complete breeds:   {breeds_complete}/{len(TARGET_BREEDS)}")
    print(f"    Incomplete breeds: {breeds_incomplete}/{len(TARGET_BREEDS)}")
    print(f"    Total clips:       {total_have}")
    print(f"    Clips still needed: ~{total_needed}")

    # Scraping progress
    progress_path = OUTPUT_DIR / "scraping_progress.json"
    if progress_path.exists():
        with open(str(progress_path)) as f:
            progress = json.load(f)
        print(f"\n  Last scraping session:")
        print(f"    Breeds processed: {len(progress)}")
        total_saved = sum(s.get("clips_saved", 0) for s in progress)
        total_errors = sum(s.get("errors", 0) for s in progress)
        print(f"    Clips saved: {total_saved}")
        print(f"    Errors: {total_errors}")

    print(f"\n{'=' * 70}")


# ============================================================================
# STEP RUNNERS
# ============================================================================
def run_train():
    """Train the bark detector model."""
    print(f"\n{'=' * 70}")
    print(f"  STEP 1: Training Bark Detector")
    print(f"{'=' * 70}")

    if BARK_MODEL_PATH.exists():
        print(f"  Model already exists at {BARK_MODEL_PATH}")
        resp = input("  Retrain? (y/n): ").strip().lower()
        if resp not in ("y", "yes"):
            print("  Skipping training.")
            return

    from bark_detector.train_bark_detector import main as train_main
    train_main()


def run_scrape(breed=None, max_videos=20, target_clips=200):
    """Run YouTube scraper for one or all breeds."""
    from bark_detector.youtube_scraper import scrape_breed, batch_scrape, check_ytdlp
    from bark_detector.model import BarkDetector

    if not check_ytdlp():
        print("ERROR: yt-dlp is not installed!")
        print("Install: pip install yt-dlp")
        return

    # Load bark detector
    bark_detector = None
    if BARK_MODEL_PATH.exists():
        print(f"  Loading bark detector model...")
        bark_detector = BarkDetector(BARK_MODEL_PATH)
        print(f"  Model loaded!")
    else:
        print(f"  WARNING: No bark detector model. Using acoustic heuristics only.")
        print(f"  For better filtering, run: python -m bark_detector.pipeline --step train")

    if breed:
        print(f"\n  Scraping breed: {breed}")
        scrape_breed(breed, max_videos=max_videos,
                     bark_detector=bark_detector, target_clips=target_clips)
    else:
        # Only scrape breeds that need data
        breeds_to_scrape = []
        for b in TARGET_BREEDS:
            breed_dir = BREED_AUDIO_DIR / b
            count = len(list(breed_dir.glob("*.wav"))) if breed_dir.exists() else 0
            if count < MIN_CLIPS_PER_BREED:
                breeds_to_scrape.append(b)

        if not breeds_to_scrape:
            print("  All target breeds have sufficient data!")
            return

        print(f"  Breeds needing data: {len(breeds_to_scrape)}")
        batch_scrape(breeds_to_scrape, max_videos_per_breed=max_videos,
                     target_clips_per_breed=target_clips)


def run_validate(breed=None, auto=True, sample_size=None):
    """Run validation on scraped clips."""
    from bark_detector.validate import validate_clips, auto_validate

    if breed:
        audio_dir = BREED_AUDIO_DIR / breed
        if auto:
            auto_validate(audio_dir)
        else:
            validate_clips(audio_dir, sample_size=sample_size)
    else:
        # Auto-validate all breeds with YouTube clips
        for b in TARGET_BREEDS:
            audio_dir = BREED_AUDIO_DIR / b
            yt_clips = list(audio_dir.glob("yt_*.wav")) if audio_dir.exists() else []
            if yt_clips:
                print(f"\n  Auto-validating {b} ({len(yt_clips)} clips)...")
                auto_validate(audio_dir)


def run_full_pipeline(max_videos=20, target_clips=200):
    """Run the complete pipeline end-to-end."""
    print(f"\n{'=' * 70}")
    print(f"  FULL PIPELINE EXECUTION")
    print(f"{'=' * 70}")

    # Step 1: Train
    if not BARK_MODEL_PATH.exists():
        print(f"\n  No bark detector model found. Training first...")
        run_train()
    else:
        print(f"\n  Bark detector model found. Skipping training.")

    # Step 2: Scrape
    print(f"\n  Starting YouTube scraping...")
    run_scrape(max_videos=max_videos, target_clips=target_clips)

    # Step 3: Auto-validate
    print(f"\n  Running auto-validation on scraped clips...")
    run_validate(auto=True)

    # Step 4: Status report
    check_status()

    print(f"\n  Pipeline complete!")
    print(f"  For manual validation: python -m bark_detector.pipeline --step validate --breed <breed> --manual")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Bark Detection Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  train         Train bark detector on existing breed audio
  scrape        Scrape YouTube for bark audio (use --breed or all)
  scrape-all    Scrape all target breeds that need data
  validate      Validate scraped clips (auto or manual)
  status        Show data collection progress
  full          Run entire pipeline (train -> scrape -> validate)

Examples:
  python -m bark_detector.pipeline --step status
  python -m bark_detector.pipeline --step train
  python -m bark_detector.pipeline --step scrape --breed beagle
  python -m bark_detector.pipeline --step scrape-all --max-videos 15
  python -m bark_detector.pipeline --step validate --breed beagle --manual
  python -m bark_detector.pipeline --step full
        """
    )
    parser.add_argument("--step", type=str, required=True,
                        choices=["train", "scrape", "scrape-all", "validate", "status", "full"],
                        help="Pipeline step to run")
    parser.add_argument("--breed", type=str, default=None,
                        help="Target breed name")
    parser.add_argument("--max-videos", type=int, default=20,
                        help="Max YouTube videos per breed (default: 20)")
    parser.add_argument("--target-clips", type=int, default=200,
                        help="Target clips per breed (default: 200)")
    parser.add_argument("--manual", action="store_true",
                        help="Use manual validation instead of auto")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample size for manual validation")

    args = parser.parse_args()

    if args.step == "status":
        check_status()
    elif args.step == "train":
        run_train()
    elif args.step == "scrape":
        run_scrape(breed=args.breed, max_videos=args.max_videos,
                   target_clips=args.target_clips)
    elif args.step == "scrape-all":
        run_scrape(max_videos=args.max_videos, target_clips=args.target_clips)
    elif args.step == "validate":
        run_validate(breed=args.breed, auto=not args.manual, sample_size=args.sample)
    elif args.step == "full":
        run_full_pipeline(max_videos=args.max_videos, target_clips=args.target_clips)


if __name__ == "__main__":
    main()
