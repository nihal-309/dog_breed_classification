"""
YouTube Bark Scraper - LIHITH's Breeds (28)
=============================================
Run this script to scrape bark audio for your assigned 28 breeds.

Usage:
    python scrape_lihith.py
    python scrape_lihith.py --max-videos 10       # fewer videos (faster)
    python scrape_lihith.py --target-clips 100     # fewer clips per breed
    python scrape_lihith.py --breed border_collie  # single breed only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bark_detector.youtube_scraper import scrape_breed, check_ytdlp
from bark_detector.model import BarkDetector

# ============================================================================
# LIHITH'S 28 BREEDS (ordered by fused dataset size)
# ============================================================================
MY_BREEDS = [
    # --- Original 10 ---
    "saint_bernard",           # fused: 345
    "cocker_spaniel",          # fused: 300
    "border_collie",           # fused: 230
    "french_bulldog",          # fused: 216
    "vizsla",                  # fused: 196
    "maltese",                 # fused: 154
    "bernese_mountain_dog",    # fused: 138
    "shetland_sheepdog",       # fused: 125
    "akita",                   # fused: akita_dog (18)
    "dalmatian",               # fused: dalmation (86)
    # --- 3 old breeds (re-scrape) ---
    "husky",                   # fused: siberian_husky (201)
    "pitbull",                 # fused: american_pit_bull_terrier (165)
    "shiba_inu",               # fused: 254
    # --- 15 new from fused dataset ---
    "scotch_terrier",          # fused: 234 (Scottish Terrier)
    "bull_mastiff",            # fused: 234
    "cairn",                   # fused: 230 (Cairn Terrier)
    "borzoi",                  # fused: 219
    "malinois",                # fused: 219 (Belgian Malinois)
    "pekinese",                # fused: 215 (Pekingese)
    "collie",                  # fused: 200
    "afghan_hound",            # fused: 171
    "american_bulldog",        # fused: 160
    "papillon",                # fused: 153
    "norwegian_elkhound",      # fused: 153
    "whippet",                 # fused: 141
    "weimaraner",              # fused: 126
    "irish_setter",            # fused: 121
    "tibetan_mastiff",         # fused: 119
]

BARK_MODEL_PATH = Path(__file__).parent / "outputs" / "bark_detector_best.pth"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lihith's Bark Scraper (28 breeds)")
    parser.add_argument("--breed", type=str, default=None,
                        help="Scrape single breed (e.g., 'border_collie')")
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
    print(f"  LIHITH's Bark Scraper")
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
