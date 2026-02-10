"""
Dog Breed Classification - Dataset Download & Setup Script
============================================================
This script helps you download and organize the Stanford Dogs dataset.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

# Try importing required packages
try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install requests tqdm kaggle")
    import requests
    from tqdm import tqdm


# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"


def create_directories():
    """Create project directory structure."""
    directories = [
        DATA_DIR,
        RAW_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "src",
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")


def check_kaggle_credentials():
    """Check if Kaggle API credentials are set up."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print("✓ Kaggle credentials found!")
        return True
    else:
        print("\n" + "="*60)
        print("KAGGLE API SETUP REQUIRED")
        print("="*60)
        print("""
To download datasets automatically, you need to set up Kaggle API:

1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token" - this downloads kaggle.json
4. Move kaggle.json to:
   - Windows: C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json
   - Linux/Mac: ~/.kaggle/kaggle.json
5. Run this script again

ALTERNATIVE: Manual Download
1. Go to https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset
2. Click "Download" button (requires Kaggle login)
3. Extract the zip file to: {raw_dir}
""".format(raw_dir=RAW_DIR))
        return False


def download_stanford_dogs_kaggle():
    """Download Stanford Dogs dataset using Kaggle API."""
    try:
        import kaggle
        
        print("\nDownloading Stanford Dogs Dataset from Kaggle...")
        print("This may take a few minutes (~788 MB)...\n")
        
        kaggle.api.dataset_download_files(
            'jessicali9530/stanford-dogs-dataset',
            path=str(RAW_DIR),
            unzip=True
        )
        print("✓ Download complete!")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return False


def download_competition_dataset():
    """Download Kaggle Competition dataset."""
    try:
        import kaggle
        
        print("\nDownloading Kaggle Competition Dataset...")
        print("This may take a few minutes (~750 MB)...\n")
        
        # Accept competition rules first (requires manual acceptance on website)
        kaggle.api.competition_download_files(
            'dog-breed-identification',
            path=str(RAW_DIR)
        )
        print("✓ Download complete!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nNote: You may need to accept competition rules on Kaggle website first.")
        print("Visit: https://www.kaggle.com/c/dog-breed-identification/rules")
        return False


def organize_stanford_dataset():
    """Organize Stanford Dogs dataset into train/val/test splits."""
    import random
    
    # Find the images directory
    images_dir = None
    for possible_path in [
        RAW_DIR / "images" / "Images",
        RAW_DIR / "Images",
        RAW_DIR / "stanford-dogs-dataset" / "images" / "Images",
    ]:
        if possible_path.exists():
            images_dir = possible_path
            break
    
    if not images_dir:
        print("✗ Could not find images directory. Please check the download.")
        print(f"  Looking in: {RAW_DIR}")
        return False
    
    print(f"\nOrganizing dataset from: {images_dir}")
    
    # Get all breed directories
    breed_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
    print(f"Found {len(breed_dirs)} breed directories")
    
    # Split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    random.seed(42)  # For reproducibility
    
    for breed_dir in tqdm(breed_dirs, desc="Processing breeds"):
        # Clean breed name (remove n02XXXXXX- prefix)
        breed_name = breed_dir.name
        if '-' in breed_name:
            breed_name = breed_name.split('-', 1)[1]
        breed_name = breed_name.replace('_', ' ').lower().replace(' ', '_')
        
        # Get all images for this breed
        images = list(breed_dir.glob("*.jpg")) + list(breed_dir.glob("*.JPEG"))
        random.shuffle(images)
        
        # Calculate split indices
        n_train = int(len(images) * TRAIN_RATIO)
        n_val = int(len(images) * VAL_RATIO)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Create breed directories and copy images
        for split_dir, split_images in [
            (TRAIN_DIR / breed_name, train_images),
            (VAL_DIR / breed_name, val_images),
            (TEST_DIR / breed_name, test_images),
        ]:
            split_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, split_dir / img_path.name)
    
    print("\n✓ Dataset organized successfully!")
    print_dataset_stats()
    return True


def print_dataset_stats():
    """Print dataset statistics."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    for split_name, split_dir in [("Train", TRAIN_DIR), ("Validation", VAL_DIR), ("Test", TEST_DIR)]:
        if split_dir.exists():
            breeds = list(split_dir.iterdir())
            total_images = sum(len(list(b.glob("*.*"))) for b in breeds if b.is_dir())
            print(f"{split_name:12} - {len(breeds):3} breeds, {total_images:5} images")
    
    print("="*60)


def main():
    print("="*60)
    print("DOG BREED CLASSIFICATION - DATASET SETUP")
    print("="*60)
    
    # Step 1: Create directories
    print("\n[1/3] Creating project directories...")
    create_directories()
    
    # Step 2: Check for existing data
    if (TRAIN_DIR / "beagle").exists():
        print("\n✓ Dataset already organized!")
        print_dataset_stats()
        return
    
    # Step 3: Check Kaggle credentials
    print("\n[2/3] Checking Kaggle credentials...")
    if check_kaggle_credentials():
        # Try downloading
        print("\n[3/3] Downloading dataset...")
        if download_stanford_dogs_kaggle():
            print("\n[4/4] Organizing dataset...")
            organize_stanford_dataset()
    else:
        print("\nPlease set up Kaggle API or download manually.")
        print("After downloading, run this script again to organize the data.")
        
        # Check if data was manually downloaded
        if any(RAW_DIR.iterdir()) if RAW_DIR.exists() else False:
            print("\nFound files in raw directory. Attempting to organize...")
            organize_stanford_dataset()


if __name__ == "__main__":
    main()
