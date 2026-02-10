# Dog Breed Bark Audio Scraper

Automated YouTube scraper that downloads bark audio and filters it using a **99.2% accurate CNN bark detector** — so only real bark sounds make it into the dataset.

## Quick Start (for Uttam & Lihith)

### 1. Clone & Install

```bash
git clone <repo-url>
cd dog_breed_classification
pip install -r requirements.txt
pip install yt-dlp
```

### 2. Run YOUR script

```bash
# Uttam runs:
python scrape_uttam.py

# Lihith runs:
python scrape_lihith.py

# Nihal runs:
python scrape_nihal.py
```

### 3. Options

```bash
# Fewer videos per breed (faster, ~3-4 hours)
python scrape_uttam.py --max-videos 10

# Fewer clips per breed
python scrape_uttam.py --target-clips 100

# Test with a single breed first
python scrape_uttam.py --breed doberman --max-videos 5
```

### 4. Share Results

After scraping, zip your `data/breed_audio/` folder and share it.
Each breed's clips will be in `data/breed_audio/<breed_name>/`.

---

## Breed Assignments

| Person | Breeds (10 each) |
|--------|-------------------|
| **Nihal** | pug, golden_retriever, bulldog, beagle, poodle, rottweiler, yorkshire_terrier, boxer, dachshund, pembroke_welsh_corgi |
| **Uttam** | doberman, great_dane, miniature_schnauzer, australian_shepherd, cavalier_king_charles, shih_tzu, boston_terrier, pomeranian, havanese, english_springer_spaniel |
| **Lihith** | shetland_sheepdog, bernese_mountain_dog, border_collie, cocker_spaniel, vizsla, french_bulldog, maltese, saint_bernard, akita, dalmatian |

---

## How the Filtering Works

Every downloaded YouTube video goes through **4 layers of filtering**:

1. **Search Optimization** — queries like `"beagle dog barking"`, `"beagle bark close up"`
2. **Segmentation** — splits video audio into 4-second clips, discards silence
3. **Acoustic Heuristic Filter** — checks spectral centroid, onset rate, ZCR, energy (rejects music/speech/noise)
4. **CNN Bark Detector** (99.2% accuracy) — trained on 7,500 real dog barks vs 6,000 non-bark samples

Only clips that pass ALL layers get saved. Expect ~30-40% of raw clips to pass.

---

## Time Estimate

- **10 breeds, 20 videos each**: ~6-10 hours
- **10 breeds, 10 videos each**: ~3-5 hours
- **Test 1 breed, 5 videos**: ~15-20 min

**Tip**: Start with `--max-videos 5 --breed <one_breed>` to test, then run the full script.

---

## Output

Clips are saved as WAV files in:
```
data/breed_audio/<breed_name>/yt_<breed>_00001_s85.wav
```
- `yt_` prefix = scraped from YouTube
- `s85` = bark confidence score (85%)

---

## Requirements

- Python 3.10+
- ~2GB free disk space
- Internet connection
- `yt-dlp` (installed via pip)
