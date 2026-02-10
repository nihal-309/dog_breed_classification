"""
Dataset Fusion Script for Dog Breed Classification
====================================================
Merges multiple dog breed datasets into a unified training set.

Handles:
- Breed name normalization (different naming conventions)
- Duplicate detection and removal
- Train/Val/Test split creation
- Class balancing information
"""

import os
import sys
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
import random

try:
    from tqdm import tqdm
    from PIL import Image
except ImportError:
    os.system(f"{sys.executable} -m pip install tqdm pillow")
    from tqdm import tqdm
    from PIL import Image


# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
ADDITIONAL_DIR = DATA_DIR / "additional_datasets"
FUSED_DIR = DATA_DIR / "fused"

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# Breed Name Normalization
# =============================================================================

# Map various breed names to a standard format
BREED_NAME_MAP = {
    # Oxford-IIIT variations
    'american_bulldog': 'american_bulldog',
    'american_pit_bull_terrier': 'american_pit_bull_terrier',
    'basset_hound': 'basset',
    'english_cocker_spaniel': 'cocker_spaniel',
    'english_setter': 'english_setter',
    'german_shorthaired': 'german_short_haired_pointer',
    'great_pyrenees': 'great_pyrenees',
    'japanese_chin': 'japanese_spaniel',
    'staffordshire_bull_terrier': 'staffordshire_bullterrier',
    'wheaten_terrier': 'soft_coated_wheaten_terrier',
    
    # Common variations
    'german shepherd': 'german_shepherd',
    'german-shepherd': 'german_shepherd',
    'golden retriever': 'golden_retriever',
    'golden-retriever': 'golden_retriever',
    'labrador retriever': 'labrador_retriever',
    'labrador-retriever': 'labrador_retriever',
    'siberian husky': 'siberian_husky',
    'siberian-husky': 'siberian_husky',
    'shih tzu': 'shih_tzu',
    'shih-tzu': 'shih_tzu',
    'pit bull': 'american_pit_bull_terrier',
    'pitbull': 'american_pit_bull_terrier',
    'bull dog': 'bulldog',
    'bull-dog': 'bulldog',
    'french bull dog': 'french_bulldog',
    'french-bulldog': 'french_bulldog',
    'yorkshire terrier': 'yorkshire_terrier',
    'yorkshire-terrier': 'yorkshire_terrier',
    'border-collie': 'border_collie',
    'border collie': 'border_collie',
}


def normalize_breed_name(name: str) -> str:
    """Normalize breed name to standard format."""
    # Convert to lowercase and replace separators
    normalized = name.lower().strip()
    normalized = normalized.replace('-', '_').replace(' ', '_')
    
    # Remove common prefixes (like n02xxxxxx- from ImageNet)
    if '_' in normalized and normalized.split('_')[0].startswith('n0'):
        normalized = '_'.join(normalized.split('_')[1:])
    
    # Check mapping
    if normalized in BREED_NAME_MAP:
        return BREED_NAME_MAP[normalized]
    
    return normalized


def get_image_hash(image_path: Path) -> str:
    """Get hash of image for duplicate detection."""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None


def is_valid_image(image_path: Path) -> bool:
    """Check if image is valid and can be opened."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False


# =============================================================================
# Dataset Collectors
# =============================================================================

def collect_stanford_dogs() -> dict:
    """Collect images from Stanford Dogs dataset."""
    print("\n[1/4] Collecting Stanford Dogs dataset...")
    
    images = defaultdict(list)
    stanford_dir = DATA_DIR / "train"
    
    if not stanford_dir.exists():
        print("  ⚠ Stanford Dogs not found. Run download_dataset.py first.")
        return images
    
    for breed_dir in stanford_dir.iterdir():
        if breed_dir.is_dir():
            breed_name = normalize_breed_name(breed_dir.name)
            for img_path in breed_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    images[breed_name].append(img_path)
    
    total = sum(len(v) for v in images.values())
    print(f"  ✓ Found {total} images in {len(images)} breeds")
    return images


def collect_oxford_pets() -> dict:
    """Collect dog images from Oxford-IIIT Pet dataset."""
    print("\n[2/4] Collecting Oxford-IIIT Pet dataset (dogs only)...")
    
    images = defaultdict(list)
    oxford_dir = ADDITIONAL_DIR / "oxford_pets" / "dogs_only"
    
    if not oxford_dir.exists():
        oxford_dir = ADDITIONAL_DIR / "oxford_pets" / "images"
        if not oxford_dir.exists():
            print("  ⚠ Oxford Pets not found. Run download_additional_datasets.py first.")
            return images
    
    # Dog breeds in Oxford dataset
    DOG_BREEDS = {
        'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle',
        'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter',
        'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin',
        'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland',
        'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier',
        'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier'
    }
    
    for item in oxford_dir.iterdir():
        if item.is_dir():
            breed_name = normalize_breed_name(item.name)
            for img_path in item.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    images[breed_name].append(img_path)
        elif item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Images directly in folder (original structure)
            filename = item.stem
            parts = filename.rsplit('_', 1)
            if len(parts) == 2:
                breed_name = parts[0].lower()
                if breed_name in DOG_BREEDS:
                    normalized = normalize_breed_name(breed_name)
                    images[normalized].append(item)
    
    total = sum(len(v) for v in images.values())
    print(f"  ✓ Found {total} images in {len(images)} breeds")
    return images


def collect_70_dog_breeds() -> dict:
    """Collect images from 70 Dog Breeds dataset."""
    print("\n[3/4] Collecting 70 Dog Breeds dataset...")
    
    images = defaultdict(list)
    dataset_dir = ADDITIONAL_DIR / "70_dog_breeds"
    
    if not dataset_dir.exists():
        print("  ⚠ 70 Dog Breeds not found. Run download_additional_datasets.py first.")
        return images
    
    # Check for train/test/valid structure
    for split in ['train', 'test', 'valid']:
        split_dir = dataset_dir / split
        if split_dir.exists():
            for breed_dir in split_dir.iterdir():
                if breed_dir.is_dir():
                    breed_name = normalize_breed_name(breed_dir.name)
                    for img_path in breed_dir.glob("*.*"):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            images[breed_name].append(img_path)
    
    total = sum(len(v) for v in images.values())
    print(f"  ✓ Found {total} images in {len(images)} breeds")
    return images


def collect_dog_breeds_darshan() -> dict:
    """Collect images from Dog Breeds (Darshan) dataset."""
    print("\n[4/5] Collecting Dog Breeds (Darshan Thakare) dataset...")
    
    images = defaultdict(list)
    dataset_dir = ADDITIONAL_DIR / "dog_breeds_darshan"
    
    if not dataset_dir.exists():
        print("  ⚠ Dog Breeds (Darshan) not found. Run download_additional_datasets.py first.")
        return images
    
    # Recursively find all breed directories
    for item in dataset_dir.rglob("*"):
        if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Get breed from parent directory
            breed_name = normalize_breed_name(item.parent.name)
            if breed_name and breed_name != 'dog_breeds_darshan':
                images[breed_name].append(item)
    
    total = sum(len(v) for v in images.values())
    print(f"  ✓ Found {total} images in {len(images)} breeds")
    return images


def collect_custom_datasets() -> dict:
    """
    Collect images from any custom datasets added to additional_datasets folder.
    
    To add a new dataset:
    1. Create folder: data/additional_datasets/<your_dataset_name>/
    2. Inside, organize images by breed:
       <your_dataset_name>/
           breed1/
               image1.jpg
               image2.jpg
           breed2/
               image1.jpg
           ...
    3. Run fuse_datasets.py again
    
    The script will automatically detect and include your new dataset!
    """
    print("\n[5/5] Scanning for custom/additional datasets...")
    
    images = defaultdict(list)
    
    # Known dataset folders to skip (already processed separately)
    KNOWN_DATASETS = {'oxford_pets', '70_dog_breeds', 'dog_breeds_darshan'}
    
    if not ADDITIONAL_DIR.exists():
        return images
    
    custom_count = 0
    for dataset_folder in ADDITIONAL_DIR.iterdir():
        if dataset_folder.is_dir() and dataset_folder.name not in KNOWN_DATASETS:
            custom_count += 1
            print(f"  → Found custom dataset: {dataset_folder.name}")
            
            # Recursively find all images
            for item in dataset_folder.rglob("*"):
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    # Get breed from parent directory
                    breed_name = normalize_breed_name(item.parent.name)
                    # Skip if parent is the dataset root folder itself
                    if breed_name and breed_name != normalize_breed_name(dataset_folder.name):
                        images[breed_name].append(item)
    
    if custom_count == 0:
        print("  ℹ No custom datasets found")
    else:
        total = sum(len(v) for v in images.values())
        print(f"  ✓ Found {total} images in {len(images)} breeds from {custom_count} custom dataset(s)")
    
    return images


# =============================================================================
# Fusion Functions
# =============================================================================

def merge_datasets(*datasets) -> dict:
    """Merge multiple dataset dictionaries."""
    print("\n" + "="*60)
    print("MERGING DATASETS")
    print("="*60)
    
    merged = defaultdict(list)
    
    for dataset in datasets:
        for breed, images in dataset.items():
            merged[breed].extend(images)
    
    print(f"✓ Total breeds: {len(merged)}")
    print(f"✓ Total images: {sum(len(v) for v in merged.values())}")
    
    return merged


def remove_duplicates(dataset: dict) -> dict:
    """Remove duplicate images based on content hash."""
    print("\n" + "="*60)
    print("REMOVING DUPLICATES")
    print("="*60)
    
    deduplicated = {}
    total_removed = 0
    seen_hashes = set()
    
    for breed, images in tqdm(dataset.items(), desc="Processing breeds"):
        unique_images = []
        
        for img_path in images:
            img_hash = get_image_hash(img_path)
            if img_hash and img_hash not in seen_hashes:
                seen_hashes.add(img_hash)
                unique_images.append(img_path)
            else:
                total_removed += 1
        
        deduplicated[breed] = unique_images
    
    print(f"✓ Removed {total_removed} duplicate images")
    print(f"✓ Remaining images: {sum(len(v) for v in deduplicated.values())}")
    
    return deduplicated


def create_splits(dataset: dict, output_dir: Path):
    """Create train/val/test splits and copy images."""
    print("\n" + "="*60)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("="*60)
    
    random.seed(RANDOM_SEED)
    
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    # Clear existing
    for d in [train_dir, val_dir, test_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    
    stats = {"train": 0, "val": 0, "test": 0}
    
    for breed, images in tqdm(dataset.items(), desc="Splitting breeds"):
        if not images:
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n_train = int(len(images) * TRAIN_RATIO)
        n_val = int(len(images) * VAL_RATIO)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy images
        for split_name, split_images, split_dir in [
            ("train", train_images, train_dir),
            ("val", val_images, val_dir),
            ("test", test_images, test_dir),
        ]:
            breed_dir = split_dir / breed
            breed_dir.mkdir(parents=True, exist_ok=True)
            
            for i, img_path in enumerate(split_images):
                # Create unique filename
                new_name = f"{breed}_{i:05d}{img_path.suffix.lower()}"
                shutil.copy2(img_path, breed_dir / new_name)
                stats[split_name] += 1
    
    print(f"\n✓ Train: {stats['train']} images")
    print(f"✓ Val:   {stats['val']} images")
    print(f"✓ Test:  {stats['test']} images")
    
    return stats


def print_class_distribution(dataset: dict):
    """Print class distribution statistics."""
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION")
    print("="*60)
    
    # Sort by count
    sorted_breeds = sorted(dataset.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\n{'Breed':<40} {'Count':>8}")
    print("-" * 50)
    
    for breed, images in sorted_breeds[:20]:
        print(f"{breed:<40} {len(images):>8}")
    
    if len(sorted_breeds) > 20:
        print(f"... and {len(sorted_breeds) - 20} more breeds")
    
    counts = [len(images) for images in dataset.values()]
    print(f"\nMin images per breed: {min(counts)}")
    print(f"Max images per breed: {max(counts)}")
    print(f"Average: {sum(counts) / len(counts):.1f}")


def main():
    print("="*60)
    print("DATASET FUSION SCRIPT")
    print("="*60)
    
    # Collect from all sources
    stanford = collect_stanford_dogs()
    oxford = collect_oxford_pets()
    seventy_breeds = collect_70_dog_breeds()
    darshan = collect_dog_breeds_darshan()
    custom = collect_custom_datasets()
    
    # Merge all datasets
    merged = merge_datasets(stanford, oxford, seventy_breeds, darshan, custom)
    
    if not merged:
        print("\n✗ No datasets found! Please run download scripts first:")
        print("  python download_dataset.py")
        print("  python download_additional_datasets.py")
        return
    
    # Remove duplicates
    deduplicated = remove_duplicates(merged)
    
    # Print distribution
    print_class_distribution(deduplicated)
    
    # Create splits
    create_splits(deduplicated, FUSED_DIR)
    
    print("\n" + "="*60)
    print("FUSION COMPLETE!")
    print("="*60)
    print(f"\nFused dataset saved to: {FUSED_DIR}")
    print("\nTo train on fused dataset:")
    print(f"  python src/train.py --data-dir {FUSED_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
