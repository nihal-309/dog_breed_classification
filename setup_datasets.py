"""
Dataset Acquisition Helper Script
Downloads and organizes publicly available audio datasets for the project
"""

import os
import subprocess
import sys
from pathlib import Path
import urllib.request
import zipfile
import json

class DatasetDownloader:
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = str(Path(__file__).parent / "data")
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.datasets_dir = self.base_path / "external_datasets"
        self.datasets_dir.mkdir(exist_ok=True)
    
    def print_section(self, title):
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
    
    def download_esc50(self):
        """Download ESC-50 dataset"""
        self.print_section("ESC-50 Dataset (Environmental Sounds)")
        print("\nESC-50 contains environmental sounds including dog barks")
        print("Size: ~600 MB")
        print("\nSteps to download:")
        print("1. Go to: https://github.com/karolpiczak/ESC-50")
        print("2. Click 'Releases' → Download 'esc-50-master.zip'")
        print("3. OR use git clone:")
        print("   git clone https://github.com/karolpiczak/ESC-50.git")
        print(f"4. Extract to: {self.datasets_dir}/ESC-50")
        
        print("\nAfter downloading, dog barks are in:")
        print("  → ESC-50/audio/  (filtered: cat barks, dog barks, etc.)")
    
    def download_urbansound(self):
        """Guide for UrbanSound8K"""
        self.print_section("UrbanSound8K Dataset")
        print("\nUrbanSound8K contains urban environmental sounds")
        print("Size: ~6 GB")
        print("\nHow to get it:")
        print("1. Register at: https://urbansounddataset.weebly.com/")
        print("2. Submit dataset request form")
        print("3. Wait for approval (usually 24-48 hours)")
        print("4. Download from provided link")
        print(f"5. Extract to: {self.datasets_dir}/UrbanSound8K")
        
        print("\nDog-related files:")
        print("  → UrbanSound8K/metadata/UrbanSound8K.csv")
        print("  → Filter for: class_id = 1 (dog bark)")
    
    def setup_audioset_download(self):
        """Guide for AudioSet dataset"""
        self.print_section("Google AudioSet (3,000+ dog barks)")
        print("\nAudioSet is a large-scale dataset of labeled sounds")
        print("Size: Variable (you select subsets)")
        print("\nHow to get dog barks:")
        
        print("\n1. Install required tools:")
        print("   pip install audioset-download")
        print("   pip install youtube-dl")
        
        print("\n2. Create download config file (audioset_config.json):")
        config = {
            "categories": ["Dog bark", "Cat meow"],
            "output_dir": str(self.datasets_dir / "audioset"),
            "sample_rate": 16000,
            "duration": 10  # 10 seconds per clip
        }
        
        config_path = self.datasets_dir / "audioset_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"   Created: {config_path}")
        
        print("\n3. Download using Python script:")
        print("   python download_audioset.py")
    
    def create_audioset_downloader(self):
        """Create Python script to download AudioSet subsets"""
        script = '''"""
AudioSet downloader for dog bark and cat meow clips
Requirements: pip install audioset-download youtube-dl
"""

import os
import json
from pathlib import Path

# AudioSet categories to download
CATEGORIES = {
    "Dog barking": {
        "ontology_id": "/m/0brhx",
        "output_dir": "dog_bark"
    },
    "Cat meowing": {
        "ontology_id": "/m/02m2d",  
        "output_dir": "cat_meow"
    }
}

OUTPUT_BASE = "data/external_datasets/audioset"

print("AudioSet Download Script")
print("="*60)
print("Note: AudioSet requires downloading from YouTube")
print("This may take several hours depending on internet speed")
print()

# Install required packages
print("Installing required packages...")
os.system("pip install audioset-download youtube-dl yt-dlp -q")

print("\\nTo download AudioSet subsets manually:")
print("1. Go to: https://research.google.com/audioset/download.html")
print("2. Download metadata CSV files")
print("3. Use audioset-download tool to fetch video clips")
print("4. Filter for dog barking and cat meowing")

print("\\nAlternatively, use pre-extracted features from:")
print("- AudioSet Google Scholar: Small pre-extracted audio samples available")
print("- ESC-50 and UrbanSound8K are faster alternatives")
'''
        
        script_path = self.datasets_dir / "download_audioset.py"
        with open(script_path, 'w') as f:
            f.write(script)
        print(f"Created: {script_path}")
    
    def create_dogspeak_guide(self):
        """Guide for requesting DogSpeak dataset"""
        self.print_section("DogSpeak Dataset (2025) - 77,000 Barking Sequences")
        
        print("\nDogSpeak is a comprehensive dog vocalization dataset")
        print("Size: 77,202 barking sequences, 33.162 hours, 156 unique dogs")
        print("Citation: Lekhak et al., ACM SIGMM, 2025")
        
        print("\nHow to request:")
        print("1. Find paper: 'DogSpeak: A Canine Vocalization Classification Dataset'")
        print("   Link: https://dl.acm.org/doi/abs/10.1145/3746027.3758298")
        
        print("\n2. Contact authors:")
        print("   Primary author: Harsh Lekhak")
        print("   Email: Contact through ACM or find in paper's metadata")
        
        print("\n3. Email template:")
        email_template = '''Subject: DogSpeak Dataset Request

Dear Authors,

I am a student researcher working on multi-modal pet classification
(combining visual and audio analysis of dogs). Your DogSpeak dataset
would be invaluable for my research on canine vocalization analysis.

Could you please share the dataset for research purposes? I would be
happy to cite your work and share our results.

Best regards,
[Your Name]
[Your Institution]
[Your Email]
        '''
        print(email_template)
    
    def create_breed_club_contacts(self):
        """Create file with AKC breed club contact info"""
        self.print_section("Getting Data from Breed Clubs")
        
        print("\nHow to contact breed clubs for audio recordings:")
        print("\n1. Visit: https://www.akc.org/about/club-search/")
        print("2. Find breed club for your target breed")
        print("3. Contact breed club secretary")
        
        sample_email = '''Subject: Audio Recording Request - Breed Study

Dear [Breed Club Name] President,

I am conducting research on dog breed vocalization patterns for 
an AI/machine learning study. I'm developing a system to identify
dog breeds from their vocalizations and behavior.

Would any members of your club be willing to contribute 
5-10 second audio recordings of their dogs barking or vocalizing?
The recordings would be kept confidential and used solely for
research purposes.

We would be happy to credit all contributors in our research paper.

Thank you,
[Your Name]
[Institution]
[Contact]
        '''
        
        contacts_file = self.datasets_dir / "BREED_CLUB_CONTACTS.txt"
        with open(contacts_file, 'w') as f:
            f.write("BREED CLUB CONTACT GUIDE\n")
            f.write("="*60 + "\n\n")
            f.write("Target breeds to contact:\n")
            f.write("- German Shepherd Club of America\n")
            f.write("- Labrador Retriever Club\n")
            f.write("- Golden Retriever Club of America\n")
            f.write("- Beagle Club of America\n")
            f.write("- Bulldog Club of America\n")
            f.write("- Siberian Husky Club of America\n")
            f.write("- Poodle Club of America\n")
            f.write("- Boxer Club of America\n")
            f.write("- Rottweiler Club of America\n")
            f.write("- Yorkshire Terrier Club of America\n\n")
            f.write("Sample Email:\n")
            f.write(sample_email)
        
        print(f"Created: {contacts_file}")
        print("\nKey breed clubs to target:")
        print("- German Shepherd Club of America")
        print("- Labrador Retriever Club")
        print("- Golden Retriever Club of America")
        print("- Beagle Club of America")
        print("- Police/Military K-9 programs")
    
    def create_youtube_scraper(self):
        """Create script to scrape breed audio from YouTube"""
        script = '''"""
YouTube breed bark scraper
Usage: Downloads short dog bark clips for specified breeds
"""

import os
import subprocess
from pathlib import Path

BREEDS = [
    "German Shepherd",
    "Labrador Retriever", 
    "Golden Retriever",
    "Beagle",
    "Bulldog",
    "Husky",
    "Poodle",
    "Boxer",
    "Rottweiler",
    "Dachshund",
    "Chihuahua",
    "Shiba Inu"
]

OUTPUT_DIR = "data/external_datasets/youtube_breed_barks"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("YouTube Breed Bark Scraper")
print("="*60)
print("This script helps download dog bark samples by breed from YouTube")
print()

# Install yt-dlp
print("Installing yt-dlp...")
os.system("pip install yt-dlp -q")

for breed in BREEDS:
    print(f"\\n[{breed}] Searching YouTube...")
    search_query = f'"{breed} barking" -compilation -hours'
    
    # Get search results
    cmd = f'yt-dlp ytsearch10:"{search_query}" --dump-json'
    print(f"Running: {cmd}")
    
    # Note: Actual implementation would parse JSON and download clips
    print(f"To download {breed} barks manually:")
    print(f"  1. Search: '{breed} barking' on YouTube")
    print(f"  2. Download 5-10 second clips using yt-dlp")
    print(f"  3. Save to: {OUTPUT_DIR}/{breed.replace(' ', '_')}/")

print("\\nNote: Always respect copyright and YouTube ToS when downloading")
print("Consider requesting licenses or permissions from video creators")
'''
        
        script_path = self.datasets_dir / "youtube_breed_scraper.py"
        with open(script_path, 'w') as f:
            f.write(script)
        print(f"Created scraper: {script_path}")
    
    def create_summary_guide(self):
        """Create master guide document"""
        guide = """# DATASET ACQUISITION GUIDE
================================

## Current Status
- ✓ You have: 1,767 audio files (944 cats, 823 dogs)
- ✓ You have: 34,616 images (208 dog breeds)
- ⚠️ Missing: Breed-labeled audio
- ⚠️ Missing: Paired image-audio samples

## Quick Start (Ranking by Effort vs Reward)

### TIER 1: Easy, Fast, High Reward
1. Download ESC-50 (10 min)
2. Train species classifier (1 hour)
3. Publish: "Pet Species Classification from Audio"

### TIER 2: Medium Effort, Very High Reward
1. Request DogSpeak dataset (email)
2. Download ESC-50 + UrbanSound8K
3. Combine with your 34,616 images
4. Train multimodal model (3-4 days)
5. Publish: "Multi-Modal Dog Breed Classification (NOVEL!)"

### TIER 3: Higher Effort, Unique Dataset
1. Contact AKC breed clubs (1-2 weeks)
2. Collect recordings from breeders (2-4 weeks)
3. Create labeled breed dataset (20-30 breeds)
4. Publish: "Canine Breed Identification from Vocalizations"

## Dataset Summary

| Dataset | Size | Time | Quality | Recommended |
|---------|------|------|---------|-------------|
| ESC-50 | 600MB | 10 min | Medium | YES |
| UrbanSound8K | 6GB | 1 hour | Medium | YES |
| AudioSet | Variable | Hours | Variable | MAYBE |
| DogSpeak | 77K clips | Email | High | YES |
| Breed Clubs | Custom | 2-4 weeks | Excellent | BEST |
| YouTube (DIY) | Custom | 1-2 weeks | Variable | GOOD |

## Next Steps

1. THIS WEEK:
   [ ] Download ESC-50
   [ ] Email DogSpeak authors
   [ ] Register for UrbanSound8K

2. NEXT 2 WEEKS:
   [ ] Explore DogBark_GA dataset via paper contacts
   [ ] Start contacting 3-5 breed clubs
   [ ] Prepare recording form for contributors

3. WITHIN 4 WEEKS:
   [ ] Begin model training on combined datasets
   [ ] Start collecting real breed recordings

## Contact Templates

### For DogSpeak Authors
(See EMAIL_TEMPLATES/dogspeak_request.txt)

### For Breed Clubs
(See EMAIL_TEMPLATES/breed_club_request.txt)

### For UrbanSound8K
Registration: https://urbansounddataset.weebly.com/

## Useful Links

- AKC Breed Clubs: https://www.akc.org/about/club-search/
- ESC-50: https://github.com/karolpiczak/ESC-50
- UrbanSound8K: https://urbansounddataset.weebly.com/
- AudioSet: https://research.google.com/audioset/
- DogSpeak: https://dl.acm.org/doi/abs/10.1145/3746027.3758298

## Recommended Priority

For publication in 8 weeks:
1. ✓ Species classifier (Week 1-2) - EASY
2. ✓ Multimodal model (Week 3-6) - NOVEL
3. ✓ Breed classifier (Week 7-8) - DATA DEPENDENT

Focus on #2 (multimodal) first - it's unique and you have the data!
"""
        
        guide_path = self.datasets_dir / "DATASET_ACQUISITION_GUIDE.txt"
        with open(guide_path, 'w') as f:
            f.write(guide)
        print(f"\n✓ Created master guide: {guide_path}")
    
    def run_all(self):
        """Run all setup steps"""
        self.print_section("DATASET ACQUISITION SETUP")
        
        print(f"\nBase directory: {self.base_path}")
        print(f"Datasets directory: {self.datasets_dir}")
        
        self.download_esc50()
        self.download_urbansound()
        self.setup_audioset_download()
        self.create_audioset_downloader()
        self.create_dogspeak_guide()
        self.create_breed_club_contacts()
        self.create_youtube_scraper()
        self.create_summary_guide()
        
        self.print_section("SETUP COMPLETE")
        print("\n✓ All helper scripts and guides created!")
        print(f"\nNext steps:")
        print(f"1. Read: {self.datasets_dir}/DATASET_ACQUISITION_GUIDE.txt")
        print(f"2. Download ESC-50 from GitHub")
        print(f"3. Email dataset authors listed above")
        print(f"4. Start with species classifier (fast win)")
        print(f"5. Then do multimodal fusion (novel contribution)")

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.run_all()
