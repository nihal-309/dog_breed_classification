# MODULES LIST AND DETAILED EXPLANATION
## Dog Breed Classification System

---

## MODULE ARCHITECTURE OVERVIEW

```
┌────────────────────────────────────────────────────────────────────┐
│                    DOG BREED CLASSIFICATION SYSTEM                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐  │
│  │   DATA MODULES      │    │    CORE MODULES                  │  │
│  ├─────────────────────┤    ├──────────────────────────────────┤  │
│  │ 1. data_loader.py   │───▶│ 3. model.py                      │  │
│  │ 2. download_*.py    │    │ 4. train.py                      │  │
│  │ 3. fuse_datasets.py │    │ 5. predict.py                    │  │
│  └─────────────────────┘    │ 6. app.py                        │  │
│          ▲                  └──────────────────────────────────┘  │
│          │                           ▲                            │
│          │                           │                            │
│          └───────────────────────────┘                            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

# PART 1: DATA MODULES

## Module 1: `data_loader.py`

### Purpose
Handles all data loading, preprocessing, and augmentation for training and inference.

### Key Functions

```python
# Function 1: get_transforms()
def get_transforms(image_size: int = 224, is_training: bool = True):
    """
    Creates data augmentation and normalization transforms.
    
    Training transforms (is_training=True):
    ├─ Random horizontal flip (50% chance)
    ├─ Random rotation (±15 degrees)
    ├─ Color jittering (brightness, contrast, saturation)
    ├─ Random crop to 224×224
    ├─ Normalize to ImageNet stats
    └─ Convert to PyTorch tensor
    
    Inference transforms (is_training=False):
    ├─ Resize to 224×224
    ├─ Center crop
    ├─ Normalize to ImageNet stats
    └─ Convert to PyTorch tensor
    """
```

**Why This Matters:**
- Training transforms prevent overfitting by exposing model to variations
- Inference transforms ensure consistent preprocessing
- ImageNet normalization enables transfer learning

```python
# Function 2: get_dataloaders()
def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
):
    """
    Creates train/val/test DataLoaders from directory structure.
    
    Directory structure expected:
    data/
    ├── train/          (70% of data)
    │   ├── breed_1/
    │   ├── breed_2/
    │   └── ...breed_120/
    ├── val/            (15% of data)
    │   └── [same structure]
    └── test/           (15% of data)
        └── [same structure]
    
    Returns:
    ├─ train_loader: Shuffled, augmented batches
    ├─ val_loader: Non-shuffled, non-augmented batches
    ├─ test_loader: Non-shuffled, non-augmented batches
    └─ class_names: List of 120 breed names
    """
```

**Key Features:**
- Automatic class discovery from directory structure
- Batch loading with shuffling for training
- No shuffling for validation/test (reproducibility)
- Parallel data loading (num_workers=4)

### Data Flow
```
Raw Images (224×224, JPEG)
    ↓
Dataloader reads batches
    ↓
Apply transforms (augmentation, normalization)
    ↓
Convert to tensors (batch_size, 3, 224, 224)
    ↓
Feed to model
```

### Code Example
```python
from src.data_loader import get_dataloaders, get_transforms

# Load data
train_loader, val_loader, test_loader, class_names = get_dataloaders(
    data_dir="data/fused",
    batch_size=32,
    image_size=224
)

# Get image from loader
images, labels = next(iter(train_loader))
print(f"Batch shape: {images.shape}")  # (32, 3, 224, 224)
print(f"Labels shape: {labels.shape}")  # (32,)
```

---

## Module 2: `download_dataset.py`

### Purpose
Automated dataset download and extraction from online sources.

### Key Functions

```python
# Downloads Stanford Dogs dataset
download_stanford_dogs(output_dir: str = "data/raw"):
    """
    Downloads Stanford Dogs dataset (20,580 images, 120 breeds)
    
    Steps:
    1. Check if already downloaded
    2. Download tar.gz file from Stanford servers
    3. Extract to data/raw/images/
    4. Download annotations (breed labels)
    5. Verify integrity (checksum)
    
    Output structure:
    data/raw/
    ├── images/
    │   ├── n02084442-pit_bull/
    │   ├── n02085620-chihuahua/
    │   └── ...
    └── annotations/
        ├── trainlists.mat
        └── testlists.mat
    """
```

### Parameters
- **output_dir:** Where to store downloaded data (default: "data/raw")
- **verify:** Check file integrity with checksums
- **extract:** Automatically extract compressed files

### Size & Time
- **Dataset Size:** ~750 MB
- **Download Time:** 5-15 minutes (depends on internet speed)
- **Extraction Time:** 2-3 minutes

### Code Example
```python
from src.download_dataset import download_stanford_dogs

# Download Stanford Dogs
download_stanford_dogs(output_dir="data/raw")

# Result: Organized breed folders in data/raw/images/
```

---

## Module 3: `fuse_datasets.py`

### Purpose
Merge multiple dog breed datasets into a unified, balanced training corpus.

### What It Does

```
Stanford Dogs          Oxford Pets         Custom Breeds
    (120 breeds)          (37 pets)          (Custom images)
        │                    │                     │
        └────────┬───────────┴──────────┬──────────┘
                 │
                 ▼
    Dataset Fusion & Harmonization
                 │
        ┌────────┼────────┐
        │        │        │
        ▼        ▼        ▼
    Train    Val      Test
    70%     15%      15%
```

### Key Functions

```python
# Function 1: fuse_datasets()
def fuse_datasets(
    datasets: List[str],
    output_dir: str = "data/fused",
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    balance: bool = True
):
    """
    Merges multiple datasets with stratified splitting.
    
    Process:
    1. Scan each dataset directory
    2. Identify breed classes (normalize names)
    3. Count images per breed
    4. Balance classes (under/over-sampling if needed)
    5. Stratified split into train/val/test (70/15/15)
    6. Copy images to unified directory structure
    7. Create metadata files
    
    Output:
    data/fused/
    ├── train/          (70% × all breeds)
    │   ├── affenpinscher/
    │   ├── afghan_hound/
    │   └── ...
    ├── val/            (15% × all breeds)
    │   └── [same structure]
    └── test/           (15% × all breeds)
        └── [same structure]
    """
```

### Balancing Strategy
```
Before Fusion:
Breed A: 500 images  ┐
Breed B: 50 images   ├─ Imbalanced!
Breed C: 200 images  ┘

After Fusion (with balancing):
Breed A: 200 images  ┐
Breed B: 200 images  ├─ Balanced!
Breed C: 200 images  ┘
(Uses oversampling for underrepresented breeds)
```

### Code Example
```python
from src.fuse_datasets import fuse_datasets

# Merge datasets
fuse_datasets(
    datasets=[
        "data/raw/stanford_dogs",
        "data/additional_datasets/oxford_pets",
        "data/additional_datasets/custom_breeds"
    ],
    output_dir="data/fused",
    train_split=0.7,
    balance=True
)
```

---

# PART 2: CORE TRAINING & INFERENCE MODULES

## Module 3: `model.py`

### Purpose
Model definitions and architecture implementations for all 7 supported CNN models.

### Supported Models

```python
def get_model(
    model_name: str = "resnet50",
    num_classes: int = 120,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Returns a model architecture for dog breed classification.
    
    Supported architectures:
    ├─ resnet50          (25.5M params, good accuracy)
    ├─ resnet101         (44.5M params, best accuracy)
    ├─ mobilenet_v2      (3.5M params, fastest)
    ├─ efficientnet_b0   (5.3M params, balanced)
    ├─ efficientnet_b3   (10.7M params, balanced+)
    ├─ vgg16             (138M params, not recommended)
    └─ densenet121       (7.9M params, balanced)
    
    Features:
    ├─ Load ImageNet pre-trained weights (optionally)
    ├─ Replace final layer for 120 dog breeds
    ├─ Option to freeze backbone (transfer learning)
    └─ Return ready-to-train model
    """
```

### Model Loading Process

```python
# Step 1: Load pre-trained backbone
weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)

# Step 2: Remove original classification head
# (keep all layers except last FC layer)

# Step 3: Add new classification head
model.fc = nn.Linear(
    in_features=model.fc.in_features,  # 2048
    out_features=num_classes  # 120 dog breeds
)

# Step 4: Optionally freeze backbone
if freeze_backbone:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():  # Unfreeze classifier
        param.requires_grad = True
```

### Helper Function: count_parameters()

```python
def count_parameters(model: nn.Module) -> dict:
    """
    Counts model parameters.
    
    Returns:
    {
        "total": 25553792,              # All parameters
        "trainable": 2050760,           # Parameters to update
        "frozen": 23503032,             # Frozen parameters
        "total_millions": 25.55,        # In millions
        "trainable_millions": 2.05      # In millions
    }
    """
```

### Code Example
```python
from src.model import get_model, count_parameters

# Create MobileNetV2 for deployment (fast)
model = get_model("mobilenet_v2", num_classes=120, pretrained=True)

# Or create ResNet50 for maximum accuracy
model = get_model("resnet50", num_classes=120, pretrained=True)

# Check parameters
params = count_parameters(model)
print(f"Total: {params['total_millions']:.2f}M")
print(f"Trainable: {params['trainable_millions']:.2f}M")
```

---

## Module 4: `train.py`

### Purpose
Training pipeline with optimization, validation, and checkpoint management.

### Key Components

```python
class Trainer:
    """
    Handles the complete training loop.
    
    Responsibilities:
    ├─ Forward pass through model
    ├─ Compute loss (CrossEntropyLoss)
    ├─ Backward pass (gradient computation)
    ├─ Update weights (optimization)
    ├─ Validation loop
    ├─ Checkpoint saving (best model)
    └─ Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 1e-4
    ):
        """Initialize trainer with model and data."""
```

### Training Configuration

```python
parser.add_argument("--model", default="resnet50",
                   choices=["resnet50", "mobilenet_v2", "efficientnet_b3", ...])
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--image-size", type=int, default=224)
parser.add_argument("--freeze-backbone", action="store_true")
```

### Training Loop Flow

```
Training Step:
    ├─ Load batch of images (32, 3, 224, 224)
    ├─ Forward pass: model(images) → logits (32, 120)
    ├─ Compute loss: CrossEntropyLoss(logits, labels)
    ├─ Backward pass: loss.backward() → gradients
    ├─ Optimizer step: optimizer.step() → update weights
    └─ Log metrics: loss, accuracy
    
Validation Step (every N batches):
    ├─ Disable gradients: torch.no_grad()
    ├─ Forward pass on validation set
    ├─ Compute accuracy, F1-score
    ├─ Compare with best validation accuracy
    ├─ Save checkpoint if improved
    └─ Adjust learning rate if no improvement
```

### Optimization Details

```python
# Optimizer: Adam (adaptive learning rates)
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4  # L2 regularization
)

# Loss function: Cross-entropy (for multi-class classification)
criterion = torch.nn.CrossEntropyLoss()

# Learning rate scheduler: Reduce on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)
```

### Output Files

```
models/resnet50_20260129_220924/
├── best_model.pth          # Best checkpoint (highest val accuracy)
├── final_model.pth         # Final model after all epochs
├── training_history.json   # Loss/accuracy curves
└── config.json            # Training configuration
```

### Code Example
```python
from src.train import main
from argparse import Namespace

# Run training
python train.py --model resnet50 --epochs 30 --batch-size 32 --lr 1e-4
```

---

## Module 5: `predict.py`

### Purpose
Make predictions on single or batch of dog images.

### Key Class: DogBreedPredictor

```python
class DogBreedPredictor:
    """
    Loads trained model and makes predictions.
    
    Workflow:
    1. Load trained model checkpoint
    2. Move to inference mode (eval)
    3. Load and preprocess image
    4. Forward pass → get probabilities
    5. Return top-K predictions with confidence
    """
    
    def __init__(
        self,
        model_path: str,
        model_name: str = "resnet50",
        device: str = None,
        class_names: list = None
    ):
        """Load model and prepare for inference."""
        self.model = get_model(model_name, ...)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()  # Inference mode
```

### Prediction Process

```
Input Image (JPEG/PNG)
    ↓
Load and convert to RGB
    ↓
Preprocess: Resize(224), Normalize
    ↓
Convert to tensor: (1, 3, 224, 224)
    ↓
Forward pass: model.forward(tensor)
    ↓
Get output logits: (1, 120)
    ↓
Apply softmax: probabilities (1, 120)
    ↓
Get top-K predictions: (breed, confidence)
    ↓
Return results [{"breed": "...", "confidence": 0.92}, ...]
```

### Methods

```python
# Method 1: Single image prediction
def predict(
    self,
    image_path: str,
    top_k: int = 5
) -> list:
    """
    Returns top-K predictions for single image.
    
    Returns:
    [
        ("golden_retriever", 0.92),
        ("labrador_retriever", 0.05),
        ("yellow_lab", 0.02),
        ...
    ]
    """

# Method 2: Batch prediction
def predict_batch(
    self,
    image_paths: list,
    top_k: int = 5
) -> list:
    """Process multiple images efficiently."""

# Method 3: Visualization
def visualize_prediction(
    self,
    image_path: str,
    top_k: int = 5,
    save_path: str = None
):
    """Display image with top-K predictions."""
```

### Code Example
```python
from src.predict import DogBreedPredictor

# Create predictor
predictor = DogBreedPredictor(
    model_path="models/resnet50_20260129_220924/best_model.pth",
    model_name="resnet50"
)

# Make prediction
results = predictor.predict(
    image_path="dog.jpg",
    top_k=5
)

for breed, confidence in results:
    print(f"{breed}: {confidence:.2%}")
```

---

## Module 6: `app.py`

### Purpose
Flask web server providing REST API and web interface for predictions.

### Architecture

```python
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Global variables for model caching
model = None
transform = None
BREED_NAMES = []
BEHAVIOR_DATA = {}
```

### Key Routes

```python
# Route 1: Web interface
@app.route('/')
def index():
    """Render main HTML page with upload form."""
    return render_template('index.html')

# Route 2: Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and return predictions.
    
    Input:
    - multipart/form-data with 'image' field (JPEG/PNG)
    
    Output:
    {
        "success": true,
        "predictions": [
            {
                "breed": "golden_retriever",
                "confidence": 0.92,
                "traits": {
                    "temperament": "Friendly, Intelligent, Devoted",
                    "size": "Large",
                    "energy_level": "High"
                }
            },
            ...
        ],
        "model_used": "resnet50",
        "inference_time_ms": 245
    }
    """
    # Validate file
    file = request.files['image']
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    # Process image
    image = Image.open(file.stream).convert("RGB")
    
    # Get predictions
    predictions = predict_breed(image, top_k=5)
    
    # Enrich with behavior data
    for pred in predictions:
        pred["traits"] = get_breed_info(pred["breed"])
    
    return jsonify({
        "success": True,
        "predictions": predictions,
        "inference_time_ms": inference_time
    })

# Route 3: Get supported breeds
@app.route('/breeds', methods=['GET'])
def get_breeds():
    """Return list of 120 supported dog breeds."""
    return jsonify({"breeds": BREED_NAMES})

# Route 4: Behavior info
@app.route('/breed/<breed_name>', methods=['GET'])
def get_breed_info(breed_name):
    """Return AKC behavior info for specific breed."""
    return jsonify(BEHAVIOR_DATA.get(breed_name, {}))

# Route 5: Model info
@app.route('/models', methods=['GET'])
def get_models():
    """Return available models."""
    return jsonify({
        "available_models": ["resnet50", "mobilenet_v2"],
        "current_model": "resnet50"
    })
```

### Startup Process

```python
if __name__ == '__main__':
    # 1. Load trained model into memory
    load_model()
    
    # 2. Load behavior database (AKC data)
    load_behavioral_data()
    
    # 3. Start Flask server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )
    
    # Server ready at http://localhost:5000
```

### Code Example
```bash
# Start the server
python app.py

# Then visit http://localhost:5000 in browser
# Upload dog image → Get predictions instantly
```

---

## MODULE DEPENDENCY DIAGRAM

```
┌────────────────────────────────────────┐
│         data_loader.py                 │
│  (Load & preprocess images)            │
└────────────┬────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│         model.py                       │
│  (Define CNN architectures)            │
└────────────┬────────────────────────────┘
             │
             ├─────────────────┬──────────────────┐
             ▼                 ▼                  ▼
      ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
      │  train.py    │  │ predict.py  │  │   app.py     │
      │  (Training)  │  │ (Inference) │  │ (Web Server) │
      └──────────────┘  └─────────────┘  └──────────────┘
             │                 │                  │
             └─────────┬───────┴──────────────────┘
                       ▼
              ┌──────────────────┐
              │   Trained Models │
              │  (*.pth files)   │
              └──────────────────┘
```

---

## MODULE USAGE FLOW

```
TRAINING PIPELINE:
1. download_dataset.py → Fetch Stanford Dogs (750MB)
2. fuse_datasets.py → Merge & balance datasets
3. data_loader.py → Load batches with augmentation
4. model.py → Create ResNet50/MobileNetV2
5. train.py → Train for 30 epochs
6. Output: best_model.pth (100MB)

INFERENCE PIPELINE:
1. app.py → Start Flask server
2. User uploads image → HTML form
3. app.py → Receives image
4. predict.py → Load model, preprocess, predict
5. data_loader.py → Apply inference transforms
6. model.py → Forward pass
7. Output: JSON with predictions + traits
```

---

## SUMMARY TABLE

| Module | Purpose | Key Classes/Functions | Input | Output |
|--------|---------|----------------------|-------|--------|
| **data_loader.py** | Data loading & preprocessing | `get_dataloaders()`, `get_transforms()` | Raw images | Batches of tensors |
| **model.py** | Model architectures | `get_model()`, `count_parameters()` | Config | PyTorch model |
| **train.py** | Training loop | `Trainer`, `main()` | Model + data | Trained checkpoint |
| **predict.py** | Inference | `DogBreedPredictor` | Image path | Top-K predictions |
| **app.py** | Web server | Flask routes | HTTP request | JSON response |
| **download_dataset.py** | Dataset download | `download_stanford_dogs()` | None | 750MB dataset |
| **fuse_datasets.py** | Dataset fusion | `fuse_datasets()` | Multiple datasets | Balanced unified dataset |

---

## KEY TECHNICAL DETAILS

### Data Flow Across Modules
```
Raw Images
   ↓ (download_dataset.py)
Organized Structure
   ↓ (fuse_datasets.py)
Balanced Train/Val/Test
   ↓ (data_loader.py)
Augmented Batches
   ↓ (model.py)
Deep Features + Classification
   ↓ (train.py)
Optimized Weights
   ↓ (predict.py / app.py)
Breed Prediction + Confidence
   ↓
User Results
```

### Module Interdependencies
- **app.py** depends on: predict.py, model.py, data_loader.py
- **predict.py** depends on: model.py, data_loader.py
- **train.py** depends on: model.py, data_loader.py
- **fuse_datasets.py** depends on: download_dataset.py
- **data_loader.py** has no dependencies (core utility)
- **model.py** has no dependencies (core utility)
