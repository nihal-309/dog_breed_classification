# System Overview

## DOG BREED CLASSIFICATION SYSTEM

### PROPOSED SYSTEM OVERVIEW

• **Multi-source Data Fusion**: Aggregate and harmonize dog breed images from multiple datasets (Stanford Dogs, Oxford Pets, custom datasets) to create a comprehensive training corpus with proper validation splits.

• **Deep Learning Classification Pipeline**: Employ transfer learning with pre-trained CNNs (ResNet50, MobileNetV2) fine-tuned on fused dataset to achieve robust breed identification across 120+ dog breeds.

• **Web-based Inference Service**: Provide real-time breed prediction through a Flask API with image preprocessing, model inference, and confidence scoring integrated with behavioral trait lookup.

• **Performance Monitoring**: Track model accuracy, inference latency, and prediction confidence distributions through comprehensive logging and metrics collection.

---

## SYSTEM ARCHITECTURE

```
┌─────────────────┐
│   User Upload   │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Image Preprocessing │
│   (Resize, Norm)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model Selection    │
│ (ResNet50/MobileNet)│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Breed Prediction    │
│  (CNN Inference)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Confidence Scoring  │
│   (Top-K breeds)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Behavior Enrichment │
│   (AKC Data Join)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Response JSON     │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Logging & Metrics   │
└─────────────────────┘
```

---

## KEY COMPONENTS

### 1. Data Pipeline
- **Dataset Download**: Automated scripts for fetching Stanford Dogs, Oxford Pets datasets
- **Data Fusion**: Merging multiple sources with consistent labeling and directory structure
- **Train/Val/Test Split**: 70/15/15 stratified split across 120+ breeds

### 2. Model Training
- **Base Models**: ResNet50 (accuracy-focused), MobileNetV2 (speed-focused)
- **Transfer Learning**: ImageNet pre-trained weights with fine-tuning
- **Data Augmentation**: Random flips, rotations, color jittering
- **Optimization**: Adam optimizer with learning rate scheduling

### 3. Inference Service
- **Flask Web Server**: RESTful API for image upload and prediction
- **Image Handler**: File validation, resizing (224×224), normalization
- **Model Loader**: Dynamic model selection based on performance requirements
- **Response Builder**: JSON output with breed, confidence, and traits

### 4. Behavioral Integration
- **AKC Database**: 200+ breeds with temperament, size, energy levels
- **Trait Mapping**: Join prediction results with behavioral characteristics
- **Enriched Output**: Provide breed name + personality traits + care requirements

---

## DATA FLOW

**Training Phase:**
1. Download raw datasets → 2. Fuse and organize → 3. Train CNN models → 4. Save checkpoints → 5. Evaluate on test set

**Inference Phase:**
1. User uploads image → 2. Preprocess image → 3. Run inference → 4. Score predictions → 5. Fetch behavior data → 6. Return results → 7. Log metrics

---

## TECHNOLOGY STACK

- **Framework**: PyTorch / TensorFlow
- **Web Server**: Flask
- **Pre-trained Models**: ResNet50, MobileNetV2
- **Data Processing**: NumPy, Pandas, PIL
- **Frontend**: HTML5, CSS3, JavaScript
