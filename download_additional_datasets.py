"""
Download Additional Datasets for Dog Breed Classification
===========================================================
Downloads and organizes multiple datasets to fuse with Stanford Dogs.

Datasets included:
1. Oxford-IIIT Pet Dataset (~800 MB)
2. 70 Dog Breeds Image Dataset (~226 MB) - Kaggle
3. Dog Breeds Image Dataset (~789 MB) - Kaggle
"""

import os
import sys
import zipfile
import tarfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve

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
ADDITIONAL_DIR = DATA_DIR / "additional_datasets"


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str = "Downloading"):
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urlretrieve(url, filename=str(output_path), reporthook=t.update_to)
    
    return output_path


def extract_archive(archive_path: Path, extract_to: Path):
    """Extract zip or tar.gz archive."""
    extract_to.mkdir(parents=True, exist_ok=True)
    
    if str(archive_path).endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif str(archive_path).endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    elif str(archive_path).endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
    
    print(f"✓ Extracted to: {extract_to}")


# =============================================================================
# Dataset 1: Oxford-IIIT Pet Dataset
# =============================================================================

def download_oxford_pets():
    """
    Download Oxford-IIIT Pet Dataset.
    Source: https://www.robots.ox.ac.uk/~vgg/data/pets/
    
    Contains 37 categories (25 dogs + 12 cats), ~200 images each.
    Includes: breed labels, head bounding boxes, segmentation masks.
    """
    print("\n" + "="*60)
    print("DOWNLOADING: Oxford-IIIT Pet Dataset")
    print("="*60)
    
    oxford_dir = ADDITIONAL_DIR / "oxford_pets"
    
    if (oxford_dir / "images").exists():
        print("✓ Oxford-IIIT Pet Dataset already downloaded!")
        return oxford_dir
    
    oxford_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for the dataset
    images_url = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
    annotations_url = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"
    
    try:
        # Download images
        print("\nDownloading images (~800 MB)...")
        images_tar = oxford_dir / "images.tar.gz"
        download_file(images_url, images_tar, "Oxford Pets Images")
        
        # Download annotations
        print("\nDownloading annotations...")
        annotations_tar = oxford_dir / "annotations.tar.gz"
        download_file(annotations_url, annotations_tar, "Oxford Pets Annotations")
        
        # Extract
        print("\nExtracting images...")
        extract_archive(images_tar, oxford_dir)
        
        print("Extracting annotations...")
        extract_archive(annotations_tar, oxford_dir)
        
        # Clean up
        images_tar.unlink()
        annotations_tar.unlink()
        
        print("✓ Oxford-IIIT Pet Dataset downloaded successfully!")
        return oxford_dir
        
    except Exception as e:
        print(f"✗ Error downloading Oxford Pets: {e}")
        print("\nManual download:")
        print(f"  1. Go to: https://www.robots.ox.ac.uk/~vgg/data/pets/")
        print(f"  2. Download images.tar.gz and annotations.tar.gz")
        print(f"  3. Extract to: {oxford_dir}")
        return None


def filter_oxford_dogs_only(oxford_dir: Path):
    """
    Filter Oxford-IIIT Pet Dataset to keep only dog breeds.
    
    Dog breeds in Oxford-IIIT (25 breeds):
    - american_bulldog, american_pit_bull_terrier, basset_hound, beagle,
    - boxer, chihuahua, english_cocker_spaniel, english_setter,
    - german_shorthaired, great_pyrenees, havanese, japanese_chin,
    - keeshond, leonberger, miniature_pinscher, newfoundland,
    - pomeranian, pug, saint_bernard, samoyed, scottish_terrier,
    - shiba_inu, staffordshire_bull_terrier, wheaten_terrier, yorkshire_terrier
    """
    DOG_BREEDS = {
        'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle',
        'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter',
        'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin',
        'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland',
        'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier',
        'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier'
    }
    
    images_dir = oxford_dir / "images"
    dogs_only_dir = oxford_dir / "dogs_only"
    
    if dogs_only_dir.exists():
        print("✓ Dog-only filter already applied!")
        return dogs_only_dir
    
    print("\nFiltering Oxford dataset to keep only dogs...")
    
    for breed in DOG_BREEDS:
        breed_dir = dogs_only_dir / breed.lower().replace(' ', '_')
        breed_dir.mkdir(parents=True, exist_ok=True)
    
    dog_count = 0
    for img_path in images_dir.glob("*.jpg"):
        # Image names are like: breed_name_123.jpg
        filename = img_path.stem
        # Extract breed name (everything except last number)
        parts = filename.rsplit('_', 1)
        if len(parts) == 2:
            breed_name = parts[0].lower()
            if breed_name in DOG_BREEDS:
                dest_dir = dogs_only_dir / breed_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest_dir / img_path.name)
                dog_count += 1
    
    print(f"✓ Filtered {dog_count} dog images into {len(DOG_BREEDS)} breeds")
    return dogs_only_dir


# =============================================================================
# Dataset 2: 70 Dog Breeds (Kaggle - gpiosenka)
# =============================================================================

def download_70_dog_breeds():
    """
    Download 70 Dog Breeds Image Dataset from Kaggle.
    Source: https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set
    
    Contains 9,347 images (7,946 train + 700 val + 700 test)
    Already preprocessed to 224x224x3
    """
    print("\n" + "="*60)
    print("DOWNLOADING: 70 Dog Breeds Image Dataset (Kaggle)")
    print("="*60)
    
    dataset_dir = ADDITIONAL_DIR / "70_dog_breeds"
    
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print("✓ 70 Dog Breeds dataset already downloaded!")
        return dataset_dir
    
    try:
        import kaggle
        
        print("\nDownloading from Kaggle (~226 MB)...")
        kaggle.api.dataset_download_files(
            'gpiosenka/70-dog-breedsimage-data-set',
            path=str(dataset_dir),
            unzip=True
        )
        print("✓ 70 Dog Breeds dataset downloaded successfully!")
        return dataset_dir
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nManual download:")
        print("  1. Go to: https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set")
        print("  2. Click 'Download' button")
        print(f"  3. Extract to: {dataset_dir}")
        return None


# =============================================================================
# Dataset 3: Dog Breeds Image Dataset (Kaggle - darshanthakare)
# =============================================================================

def download_dog_breeds_darshan():
    """
    Download Dog Breeds Image Dataset from Kaggle.
    Source: https://www.kaggle.com/datasets/darshanthakare/dog-breeds-image-dataset
    
    Contains 17,498 images across multiple breeds.
    """
    print("\n" + "="*60)
    print("DOWNLOADING: Dog Breeds Image Dataset (Darshan Thakare)")
    print("="*60)
    
    dataset_dir = ADDITIONAL_DIR / "dog_breeds_darshan"
    
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print("✓ Dog Breeds (Darshan) dataset already downloaded!")
        return dataset_dir
    
    try:
        import kaggle
        
        print("\nDownloading from Kaggle (~789 MB)...")
        kaggle.api.dataset_download_files(
            'darshanthakare/dog-breeds-image-dataset',
            path=str(dataset_dir),
            unzip=True
        )
        print("✓ Dog Breeds (Darshan) dataset downloaded successfully!")
        return dataset_dir
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nManual download:")
        print("  1. Go to: https://www.kaggle.com/datasets/darshanthakare/dog-breeds-image-dataset")
        print("  2. Click 'Download' button")
        print(f"  3. Extract to: {dataset_dir}")
        return None


# =============================================================================
# Utility Functions
# =============================================================================

def check_kaggle_credentials():
    """Check if Kaggle API credentials are set up."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print("✓ Kaggle credentials found!")
        return True
    
    # Check environment variable
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        print("✓ Kaggle credentials found in environment!")
        return True
    
    print("✗ Kaggle credentials not found!")
    print("\nTo set up Kaggle API:")
    print("  1. Go to https://www.kaggle.com/settings")
    print("  2. Scroll to 'API' section")
    print("  3. Click 'Create New Token'")
    print(f"  4. Move kaggle.json to: {kaggle_dir}")
    return False


def print_dataset_summary():
    """Print summary of all downloaded datasets."""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    datasets = [
        ("Stanford Dogs", DATA_DIR / "train"),
        ("Oxford Pets (Dogs)", ADDITIONAL_DIR / "oxford_pets" / "dogs_only"),
        ("70 Dog Breeds", ADDITIONAL_DIR / "70_dog_breeds"),
        ("Dog Breeds (Darshan)", ADDITIONAL_DIR / "dog_breeds_darshan"),
    ]
    
    total_images = 0
    total_breeds = set()
    
    for name, path in datasets:
        if path.exists():
            # Count images
            images = list(path.rglob("*.jpg")) + list(path.rglob("*.jpeg")) + list(path.rglob("*.png"))
            
            # Count breed folders
            breeds = [d.name for d in path.iterdir() if d.is_dir()]
            
            print(f"\n{name}:")
            print(f"  Path: {path}")
            print(f"  Images: {len(images)}")
            print(f"  Breeds/Folders: {len(breeds)}")
            
            total_images += len(images)
            total_breeds.update(breeds)
        else:
            print(f"\n{name}: Not downloaded")
    
    print("\n" + "-"*60)
    print(f"TOTAL IMAGES: {total_images}")
    print(f"UNIQUE BREED FOLDERS: {len(total_breeds)}")
    print("="*60)


def main():
    print("="*60)
    print("ADDITIONAL DATASETS DOWNLOADER")
    print("For Dog Breed Classification Project")
    print("="*60)
    
    # Create directories
    ADDITIONAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check Kaggle credentials for Kaggle datasets
    kaggle_ready = check_kaggle_credentials()
    
    # Download Oxford-IIIT Pet Dataset (direct download, no Kaggle needed)
    oxford_dir = download_oxford_pets()
    if oxford_dir:
        filter_oxford_dogs_only(oxford_dir)
    
    # Download Kaggle datasets
    if kaggle_ready:
        download_70_dog_breeds()
        download_dog_breeds_darshan()
    else:
        print("\n⚠ Skipping Kaggle datasets - credentials not configured")
        print("  Set up Kaggle API to download additional datasets")
    
    # Print summary
    print_dataset_summary()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Run 'python fuse_datasets.py' to merge all datasets")
    print("2. Run 'python src/train.py' to train on combined dataset")
    print("="*60)


if __name__ == "__main__":
    main()
