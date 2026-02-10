"""
Script to organize audio files into cats/dogs train/val folders
"""
import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
BASE_DIR = Path(r"c:\Users\Nihal\Desktop\DAA\question_papers\dog_breed_classification\data\audio_datasets")
OUTPUT_DIR = Path(r"c:\Users\Nihal\Desktop\DAA\question_papers\dog_breed_classification\data\audio_organized")

# Create output directories
output_folders = [
    OUTPUT_DIR / "train" / "cats",
    OUTPUT_DIR / "train" / "dogs",
    OUTPUT_DIR / "val" / "cats",
    OUTPUT_DIR / "val" / "dogs",
]

for folder in output_folders:
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Created: {folder}")

# Audio extensions to look for
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

# Collect all audio files
cat_files = []
dog_files = []

def classify_file(filepath):
    """Classify file as cat or dog based on filename or parent folder"""
    filename = filepath.name.lower()
    parent = filepath.parent.name.lower()
    grandparent = filepath.parent.parent.name.lower() if filepath.parent.parent else ""
    
    # Check filename patterns
    if 'cat' in filename or 'meow' in filename:
        return 'cat'
    elif 'dog' in filename or 'bark' in filename or 'growl' in filename or 'grunt' in filename:
        return 'dog'
    
    # Check parent folder names
    if 'cat' in parent or 'meow' in parent:
        return 'cat'
    elif 'dog' in parent or 'bark' in parent or 'growl' in parent or 'grunt' in parent:
        return 'dog'
    
    # Check grandparent folder names
    if 'cat' in grandparent:
        return 'cat'
    elif 'dog' in grandparent:
        return 'dog'
    
    return None

# Walk through all directories
print("\nScanning for audio files...")
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        filepath = Path(root) / file
        if filepath.suffix.lower() in AUDIO_EXTENSIONS:
            classification = classify_file(filepath)
            if classification == 'cat':
                cat_files.append(filepath)
            elif classification == 'dog':
                dog_files.append(filepath)
            else:
                print(f"Could not classify: {filepath}")

print(f"\nFound {len(cat_files)} cat audio files")
print(f"Found {len(dog_files)} dog audio files")
print(f"Total: {len(cat_files) + len(dog_files)} files")

# Shuffle files
random.shuffle(cat_files)
random.shuffle(dog_files)

# Split 80/20
def split_files(files, train_ratio=0.8):
    split_idx = int(len(files) * train_ratio)
    return files[:split_idx], files[split_idx:]

cat_train, cat_val = split_files(cat_files)
dog_train, dog_val = split_files(dog_files)

print(f"\nSplit results:")
print(f"  Cats - Train: {len(cat_train)}, Val: {len(cat_val)}")
print(f"  Dogs - Train: {len(dog_train)}, Val: {len(dog_val)}")

# Copy files to organized folders
def copy_files(file_list, dest_folder, prefix):
    """Copy files with unique naming to avoid duplicates"""
    for i, src_file in enumerate(file_list, 1):
        # Create unique filename: prefix_index_originalname
        new_name = f"{prefix}_{i:04d}{src_file.suffix}"
        dest_path = dest_folder / new_name
        shutil.copy2(src_file, dest_path)
    print(f"  Copied {len(file_list)} files to {dest_folder}")

print("\nCopying files...")
copy_files(cat_train, OUTPUT_DIR / "train" / "cats", "cat")
copy_files(cat_val, OUTPUT_DIR / "val" / "cats", "cat")
copy_files(dog_train, OUTPUT_DIR / "train" / "dogs", "dog")
copy_files(dog_val, OUTPUT_DIR / "val" / "dogs", "dog")

# Print final summary
print("\n" + "="*60)
print("ORGANIZATION COMPLETE!")
print("="*60)
print(f"\nOutput Directory: {OUTPUT_DIR}")
print(f"\nFolder Structure:")
print(f"  audio_organized/")
print(f"  ├── train/")
print(f"  │   ├── cats/  ({len(cat_train)} files)")
print(f"  │   └── dogs/  ({len(dog_train)} files)")
print(f"  └── val/")
print(f"      ├── cats/  ({len(cat_val)} files)")
print(f"      └── dogs/  ({len(dog_val)} files)")
print(f"\nTotal files organized: {len(cat_train) + len(cat_val) + len(dog_train) + len(dog_val)}")
