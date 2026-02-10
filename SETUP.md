# Setup Instructions for Uttam & Lihith

Follow these steps to get the bark scraper running on your machine.

## Prerequisites

- **Python 3.10+** — [Download](https://www.python.org/downloads/) (check "Add to PATH")
- **Git** — [Download](https://git-scm.com/download/win)
- **Internet connection** (for YouTube scraping)
- **~2GB free disk space**

## Step 1: Clone the Repository

Open PowerShell and run:

```powershell
git clone https://github.com/nihal-309/dog_breed_classification.git
cd dog_breed_classification
```

## Step 2: Create Virtual Environment (Recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you get a permission error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

## Step 3: Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `torch` — deep learning framework
- `numpy`, `scipy` — audio processing
- `yt-dlp` — YouTube audio downloading
- `imageio-ffmpeg` — bundled ffmpeg binary for audio conversion
- `matplotlib` — visualization
- `scikit-learn` — ML utilities
- `yt-dlp` — YouTube downloader
- `sounddevice` (optional) — audio playback for validation

## Step 4: Verify Setup

```powershell
python -c "import torch; import yt_dlp; print('All set!')"
```

If no errors, you're ready to scrape!

## Step 5: Run YOUR Script

### For Uttam:
```powershell
python scrape_uttam.py
```

### For Lihith:
```powershell
python scrape_lihith.py
```

### Options:

```powershell
# Test with one breed first (recommended)
python scrape_uttam.py --breed doberman --max-videos 5

# Fewer videos (faster, ~3-5 hours)
python scrape_uttam.py --max-videos 10

# Fewer clips per breed
python scrape_uttam.py --target-clips 100

# Single breed only
python scrape_uttam.py --breed doberman
```

## What You'll See

```
============================================================
  UTTAM's Bark Scraper
  Breeds: 10
  Max videos/breed: 20
  Target clips/breed: 200
============================================================

[1/10] Scraping: doberman
  Searching YouTube for 'doberman dog barking' videos...
    Found 20 unique videos
  
  Step 2: Downloading and filtering...
    [1/20] https://www.youtube.com/watch?v=...
      Title: ANGRY DOBERMAN BARKING COMPILATION
      Duration: 180s
      Segmented into 45 clips
      Filtering: 45 -> acoustic:32 -> cnn:28 -> final:24
      [SAVED 24 clips]
```

## Output Location

Downloaded clips go to:
```
data/breed_audio/<breed_name>/yt_<breed>_00001_s85.wav
```

After scraping your 10 breeds, you'll have ~1500-2000 clips total in `data/breed_audio/`.

## Sharing Results

When done, zip and share:
```powershell
# Compress your breed audio
Compress-Archive -Path data/breed_audio -DestinationPath uttam_breed_audio.zip
```

Send `uttam_breed_audio.zip` to Nihal so he can merge all results.

## Troubleshooting

### `yt-dlp: command not found`
```powershell
pip install --upgrade yt-dlp
```

### Videos not found / Request Denied
- YouTube may rate-limit your requests
- Wait 5-10 minutes and try again
- Try fewer videos: `--max-videos 10`

### Low accuracy / Mostly rejected clips
- The acoustic filter / CNN is being strict (good for quality)
- Try `--acoustic-threshold 0.3` to be more permissive
- But you'll get more false positives

### Audio playback not working
```powershell
pip install sounddevice
```

### Out of disk space
Reduce `--target-clips`:
```powershell
python scrape_uttam.py --target-clips 50
```

## Time Estimates

- **Test run** (1 breed, 5 videos): ~15-20 min
- **Quick run** (10 breeds, 10 videos each): ~3-5 hours
- **Full run** (10 breeds, 20 videos each): ~6-10 hours

---

**Questions?** Message Nihal or check [SCRAPING_README.md](SCRAPING_README.md) for more details.
