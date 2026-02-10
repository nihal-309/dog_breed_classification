"""
YouTube Bark Scraper - NIHAL's Breeds (1-10)
=============================================
Run this script to scrape bark audio for your assigned 10 breeds.

Usage:
    python scrape_nihal.py
    python scrape_nihal.py --max-videos 10       # fewer videos (faster)
    python scrape_nihal.py --target-clips 100     # fewer clips per breed
    python scrape_nihal.py --breed beagle         # single breed only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bark_detector.youtube_scraper import scrape_breed, check_ytdlp
from bark_detector.model import BarkDetector

# ============================================================================
# NIHAL'S 10 BREEDS
# ============================================================================
MY_BREEDS = [
    "pug",
    "golden_retriever",
    "bulldog",
    "beagle",
    "poodle",
    "rottweiler",
    "yorkshire_terrier",
    "boxer",
    "dachshund",
    "pembroke_welsh_corgi",
]

BARK_MODEL_PATH = Path(__file__).parent / "outputs" / "bark_detector_best.pth"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nihal's Bark Scraper (10 breeds)")
    parser.add_argument("--breed", type=str, default=None,
                        help="Scrape single breed (e.g., 'beagle')")
    parser.add_argument("--max-videos", type=int, default=20,
                        help="Max YouTube videos per breed (default: 20)")
    parser.add_argument("--target-clips", type=int, default=200,
                        help="Target clips per breed (default: 200)")
    args = parser.parse_args()

    # Check yt-dlp
    if not check_ytdlp():
        print("ERROR: yt-dlp not installed! Run: pip install yt-dlp")
        sys.exit(1)

    # Load bark detector
    bark_detector = None
    if BARK_MODEL_PATH.exists():
        print("Loading bark detector model...")
        bark_detector = BarkDetector(BARK_MODEL_PATH)
        print("Model loaded! (99.2% accuracy)")
    else:
        print("WARNING: No bark detector model found at outputs/bark_detector_best.pth")
        print("Using acoustic heuristics only (less accurate)")

    # Determine breeds to scrape
    breeds = [args.breed] if args.breed else MY_BREEDS

    print(f"\n{'=' * 60}")
    print(f"  NIHAL's Bark Scraper")
    print(f"  Breeds: {len(breeds)}")
    print(f"  Max videos/breed: {args.max_videos}")
    print(f"  Target clips/breed: {args.target_clips}")
    print(f"{'=' * 60}")

    for i, breed in enumerate(breeds, 1):
        print(f"\n[{i}/{len(breeds)}] Scraping: {breed}")
        scrape_breed(
            breed,
            max_videos=args.max_videos,
            bark_detector=bark_detector,
            target_clips=args.target_clips,
        )

    print(f"\n{'=' * 60}")
    print(f"  ALL DONE! Check data/breed_audio/ for results.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
