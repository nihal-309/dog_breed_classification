"""
YouTube Bark Scraper - UTTAM's Breeds (28)
============================================
Run this script to scrape bark audio for your assigned 28 breeds.

Usage:
    python scrape_uttam.py
    python scrape_uttam.py --max-videos 10       # fewer videos (faster)
    python scrape_uttam.py --target-clips 100     # fewer clips per breed
    python scrape_uttam.py --breed doberman       # single breed only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bark_detector.youtube_scraper import scrape_breed, check_ytdlp
from bark_detector.model import BarkDetector

# ============================================================================
# UTTAM'S 28 BREEDS (ordered by fused dataset size)
# ============================================================================
MY_BREEDS = [
    # --- Original 10 ---
    "pomeranian",              # fused: 473
    "shih_tzu",                # fused: 309
    "great_dane",              # fused: 245
    "doberman",                # fused: 229
    "havanese",                # fused: 164
    "english_springer_spaniel",# close to fused: english_springer (123)
    "miniature_schnauzer",     # fused: 98
    "boston_terrier",          # fused: 79
    "australian_shepherd",     # popular breed, good YouTube presence
    "cavalier_king_charles",   # popular breed, good YouTube presence
    # --- 3 old breeds (re-scrape) ---
    "chihuahua",               # fused: 384
    "german_shepherd",         # fused: 96
    "labrador",                # fused: 189
    # --- 15 new from fused dataset ---
    "basset",                  # fused: 439
    "newfoundland",            # fused: 384
    "samoyed",                 # fused: 328
    "miniature_pinscher",      # fused: 304
    "great_pyrenees",          # fused: 296
    "lhasa",                   # fused: 295 (Lhasa Apso)
    "english_setter",          # fused: 286
    "keeshond",                # fused: 280
    "soft_coated_wheaten_terrier",  # fused: 280
    "staffordshire_bullterrier",    # fused: 273
    "airedale",                # fused: 264 (Airedale Terrier)
    "chow",                    # fused: 258 (Chow Chow)
    "german_short_haired_pointer", # fused: 256
    "bloodhound",              # fused: 252
    "irish_wolfhound",         # fused: 245
]

BARK_MODEL_PATH = Path(__file__).parent / "outputs" / "bark_detector_best.pth"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Uttam's Bark Scraper (28 breeds)")
    parser.add_argument("--breed", type=str, default=None,
                        help="Scrape single breed (e.g., 'doberman')")
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
    print(f"  UTTAM's Bark Scraper")
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
