"""
Download Additional Dog Breed Datasets
=======================================
Downloads comprehensive behavioral, physical, and intelligence datasets
to improve breed prediction accuracy.

Recommended Datasets (from Kaggle):
1. Dog Breeds (AKC official) - breed_traits.csv with 16 traits
2. Dogs Intelligence and Size - obedience & learning ability  
3. Dog Breeds Ranking - lifetime cost, health, suitability
4. Dog Breeds Details - comprehensive physical + behavioral

Run: python download_behavior_datasets.py
"""

import os
import subprocess
import sys

# Dataset URLs and info
DATASETS = {
    "akc_breed_traits": {
        "kaggle_id": "sujaykapadnis/dog-breeds",
        "description": "AKC Official - 16 behavioral traits (1-5 scale) for 195 breeds",
        "files": ["breed_traits.csv", "breed_rank.csv"]
    },
    "intelligence_size": {
        "kaggle_id": "thedevastator/canine-intelligence-and-size",
        "description": "Intelligence scores, obedience probability, height/weight",
        "files": ["dog_intelligence.csv", "AKC Breed Info.csv"]
    },
    "dog_ranking": {
        "kaggle_id": "jainaru/dog-breeds-ranking-best-to-worst",
        "description": "87 breeds with cost, health issues, child suitability, intelligence rank",
        "files": ["dogs-ranking-dataset.csv"]
    },
    "breed_details": {
        "kaggle_id": "warcoder/dog-breeds-details",
        "description": "Physical attributes and care requirements",
        "files": ["Dog Breeds.csv"]
    }
}

def download_datasets(output_dir: str = "data/behaviour/additional"):
    """Download all recommended datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("📥 Downloading Comprehensive Dog Breed Datasets")
    print("=" * 60)
    
    # Check if kaggle is installed
    try:
        import kaggle
        print("✅ Kaggle API found")
    except ImportError:
        print("❌ Kaggle not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"])
        print("\n⚠️  Please set up your Kaggle API credentials:")
        print("   1. Go to https://www.kaggle.com/settings")
        print("   2. Click 'Create New Token' under API section")
        print("   3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)")
        return
    
    for name, info in DATASETS.items():
        print(f"\n📦 Downloading: {name}")
        print(f"   Description: {info['description']}")
        
        dataset_dir = os.path.join(output_dir, name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        try:
            cmd = f"kaggle datasets download -d {info['kaggle_id']} -p {dataset_dir} --unzip"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ Downloaded to {dataset_dir}")
            else:
                print(f"   ❌ Error: {result.stderr}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("📊 Download Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Run: python merge_behavior_datasets.py")
    print("2. This will create a unified dataset with all traits")
    print("=" * 60)


def print_manual_download_instructions():
    """Print manual download links if Kaggle API isn't set up."""
    print("\n" + "=" * 60)
    print("📥 RECOMMENDED DATASETS FOR BETTER PREDICTIONS")
    print("=" * 60)
    
    print("\n🔗 Download these datasets from Kaggle:\n")
    
    print("1️⃣  AKC Dog Breeds (BEST - Official AKC Data)")
    print("   URL: https://www.kaggle.com/datasets/sujaykapadnis/dog-breeds")
    print("   Contains: 195 breeds, 16 behavioral traits (1-5 scale)")
    print("   Features: Affectionate, Good with Kids, Energy, Trainability,")
    print("             Barking, Shedding, Drooling, Coat Type, etc.")
    
    print("\n2️⃣  Dogs Intelligence and Size")
    print("   URL: https://www.kaggle.com/datasets/thedevastator/canine-intelligence-and-size")
    print("   Contains: Obedience scores, command repetitions, height/weight")
    print("   Based on: Stanley Coren's research + AKC size data")
    
    print("\n3️⃣  Dog Breeds Ranking (Best to Worst)")
    print("   URL: https://www.kaggle.com/datasets/jainaru/dog-breeds-ranking-best-to-worst")
    print("   Contains: 87 breeds with lifetime cost, genetic ailments,")
    print("             child suitability, intelligence rank, longevity")
    
    print("\n4️⃣  Dog Breeds Details")
    print("   URL: https://www.kaggle.com/datasets/warcoder/dog-breeds-details")
    print("   Contains: Physical attributes, grooming needs, exercise needs")
    
    print("\n" + "=" * 60)
    print("📁 After downloading, place CSV files in:")
    print("   data/behaviour/additional/")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download dog breed datasets")
    parser.add_argument("--manual", action="store_true", help="Show manual download links")
    args = parser.parse_args()
    
    if args.manual:
        print_manual_download_instructions()
    else:
        try:
            download_datasets()
        except Exception as e:
            print(f"Auto-download failed: {e}")
            print_manual_download_instructions()
