# Dog Breed Classification Project

## Overview
A deep learning project to classify dog breeds from images using Convolutional Neural Networks (CNNs).

## Datasets

### 1. Stanford Dogs Dataset (Recommended to Start)
- **Images:** 20,580
- **Breeds:** 120
- **Size:** ~788 MB
- **Includes:** Class labels, Bounding boxes
- **Download:** https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

### 2. Kaggle Dog Breed Identification Competition
- **Images:** 20,581 (10,222 train + 10,357 test)
- **Breeds:** 120
- **Size:** ~750 MB
- **Download:** https://www.kaggle.com/c/dog-breed-identification/data

### 3. Oxford-IIIT Pet Dataset (Alternative)
- **Images:** 7,349
- **Classes:** 37 (25 dogs + 12 cats)
- **Download:** https://www.robots.ox.ac.uk/~vgg/data/pets/

## Project Structure
```
dog_breed_classification/
├── data/
│   ├── raw/                 # Original downloaded data
│   ├── processed/           # Preprocessed images
│   ├── train/               # Training images by breed
│   ├── val/                 # Validation images
│   └── test/                # Test images
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── models/                  # Saved model weights
├── requirements.txt
├── download_dataset.py
└── README.md
```

## Setup Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
Option A - Using Kaggle API:
```bash
# Install Kaggle API
pip install kaggle

# Set up API credentials (download kaggle.json from Kaggle account settings)
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Download Stanford Dogs
kaggle datasets download -d jessicali9530/stanford-dogs-dataset -p data/raw

# OR Download Competition Dataset
kaggle competitions download -c dog-breed-identification -p data/raw
```

Option B - Manual Download:
1. Go to https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset
2. Click "Download" button
3. Extract to `data/raw/` folder

### Step 3: Run Data Preparation
```bash
python download_dataset.py
```

## Breed List (120 breeds)
```
affenpinscher, afghan_hound, african_hunting_dog, airedale, 
american_staffordshire_terrier, appenzeller, australian_terrier, basenji, 
basset, beagle, bedlington_terrier, bernese_mountain_dog, 
black-and-tan_coonhound, blenheim_spaniel, bloodhound, bluetick, 
border_collie, border_terrier, borzoi, boston_bull, bouvier_des_flandres, 
boxer, briard, brittany_spaniel, bull_mastiff, cairn, cardigan, 
chesapeake_bay_retriever, chihuahua, chow, clumber, cocker_spaniel, collie, 
curly-coated_retriever, dandie_dinmont, dhole, dingo, doberman, 
english_foxhound, english_setter, english_springer, entlebucher, eskimo_dog, 
flat-coated_retriever, french_bulldog, german_shepherd, 
german_short-haired_pointer, giant_schnauzer, golden_retriever, gordon_setter, 
great_dane, great_pyrenees, greater_swiss_mountain_dog, groenendael, 
ibizan_hound, irish_setter, irish_terrier, irish_water_spaniel, 
irish_wolfhound, italian_greyhound, japanese_spaniel, keeshond, kelpie, 
kerry_blue_terrier, komondor, kuvasz, labrador_retriever, lakeland_terrier, 
leonberg, lhasa, malamute, malinois, maltese_dog, mexican_hairless, 
miniature_pinscher, miniature_poodle, miniature_schnauzer, newfoundland, 
norfolk_terrier, norwegian_elkhound, norwich_terrier, old_english_sheepdog, 
otterhound, papillon, pekinese, pembroke, pomeranian, pug, redbone, 
rhodesian_ridgeback, rottweiler, saint_bernard, saluki, samoyed, schipperke, 
scotch_terrier, scottish_deerhound, sealyham_terrier, shetland_sheepdog, 
shih-tzu, siberian_husky, silky_terrier, soft-coated_wheaten_terrier, 
staffordshire_bullterrier, standard_poodle, standard_schnauzer, sussex_spaniel, 
tibetan_mastiff, tibetan_terrier, toy_poodle, toy_terrier, vizsla, 
walker_hound, weimaraner, welsh_springer_spaniel, west_highland_white_terrier, 
whippet, wire-haired_fox_terrier, yorkshire_terrier
```

## Model Architectures to Try
1. **ResNet50** - Good baseline, transfer learning friendly
2. **EfficientNet** - State-of-the-art accuracy/efficiency trade-off
3. **VGG16/19** - Simple and effective
4. **MobileNetV2** - Lightweight for deployment
5. **Vision Transformer (ViT)** - Latest transformer-based approach

## References
- Khosla, A., et al. "Novel Dataset for Fine-Grained Image Categorization." FGVC Workshop, CVPR 2011.
- ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
