"""
Prediction/Inference Script for Dog Breed Classification
==========================================================
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import get_transforms
from src.model import get_model


# Breed names (120 classes from Stanford Dogs)
BREED_NAMES = [
    "affenpinscher", "afghan_hound", "african_hunting_dog", "airedale",
    "american_staffordshire_terrier", "appenzeller", "australian_terrier",
    "basenji", "basset", "beagle", "bedlington_terrier", "bernese_mountain_dog",
    "black_and_tan_coonhound", "blenheim_spaniel", "bloodhound", "bluetick",
    "border_collie", "border_terrier", "borzoi", "boston_bull",
    "bouvier_des_flandres", "boxer", "briard", "brittany_spaniel", "bull_mastiff",
    "cairn", "cardigan", "chesapeake_bay_retriever", "chihuahua", "chow",
    "clumber", "cocker_spaniel", "collie", "curly_coated_retriever",
    "dandie_dinmont", "dhole", "dingo", "doberman", "english_foxhound",
    "english_setter", "english_springer", "entlebucher", "eskimo_dog",
    "flat_coated_retriever", "french_bulldog", "german_shepherd",
    "german_short_haired_pointer", "giant_schnauzer", "golden_retriever",
    "gordon_setter", "great_dane", "great_pyrenees", "greater_swiss_mountain_dog",
    "groenendael", "ibizan_hound", "irish_setter", "irish_terrier",
    "irish_water_spaniel", "irish_wolfhound", "italian_greyhound",
    "japanese_spaniel", "keeshond", "kelpie", "kerry_blue_terrier", "komondor",
    "kuvasz", "labrador_retriever", "lakeland_terrier", "leonberg", "lhasa",
    "malamute", "malinois", "maltese_dog", "mexican_hairless", "miniature_pinscher",
    "miniature_poodle", "miniature_schnauzer", "newfoundland", "norfolk_terrier",
    "norwegian_elkhound", "norwich_terrier", "old_english_sheepdog", "otterhound",
    "papillon", "pekinese", "pembroke", "pomeranian", "pug", "redbone",
    "rhodesian_ridgeback", "rottweiler", "saint_bernard", "saluki", "samoyed",
    "schipperke", "scotch_terrier", "scottish_deerhound", "sealyham_terrier",
    "shetland_sheepdog", "shih_tzu", "siberian_husky", "silky_terrier",
    "soft_coated_wheaten_terrier", "staffordshire_bullterrier", "standard_poodle",
    "standard_schnauzer", "sussex_spaniel", "tibetan_mastiff", "tibetan_terrier",
    "toy_poodle", "toy_terrier", "vizsla", "walker_hound", "weimaraner",
    "welsh_springer_spaniel", "west_highland_white_terrier", "whippet",
    "wire_haired_fox_terrier", "yorkshire_terrier"
]


class DogBreedPredictor:
    """Dog breed prediction from images."""
    
    def __init__(
        self,
        model_path: str,
        model_name: str = "resnet50",
        device: str = None,
        class_names: list = None
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model checkpoint
            model_name: Architecture name
            device: Device to use (auto-detected if None)
            class_names: List of class names (uses default if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or BREED_NAMES
        
        # Load model
        self.model = get_model(model_name, num_classes=len(self.class_names), pretrained=False)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_transforms(image_size=224, is_training=False)
        
        print(f"Model loaded from: {model_path}")
        print(f"Using device: {self.device}")
    
    def predict(self, image_path: str, top_k: int = 5) -> list:
        """
        Predict breed for a single image.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
        
        Returns:
            List of (breed_name, probability) tuples
        """
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            breed_name = self.class_names[idx].replace("_", " ").title()
            results.append((breed_name, float(prob)))
        
        return results
    
    def predict_batch(self, image_paths: list, top_k: int = 5) -> list:
        """
        Predict breeds for multiple images.
        
        Args:
            image_paths: List of image file paths
            top_k: Number of top predictions per image
        
        Returns:
            List of prediction results for each image
        """
        results = []
        for path in image_paths:
            try:
                preds = self.predict(path, top_k)
                results.append({"path": path, "predictions": preds})
            except Exception as e:
                results.append({"path": path, "error": str(e)})
        return results
    
    def visualize_prediction(self, image_path: str, top_k: int = 5, save_path: str = None):
        """
        Visualize prediction with image and bar chart.
        
        Args:
            image_path: Path to image
            top_k: Number of predictions to show
            save_path: Path to save figure (displays if None)
        """
        predictions = self.predict(image_path, top_k)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        image = Image.open(image_path)
        ax1.imshow(image)
        ax1.set_title(f"Predicted: {predictions[0][0]}")
        ax1.axis("off")
        
        # Show predictions bar chart
        breeds = [p[0] for p in predictions]
        probs = [p[1] * 100 for p in predictions]
        colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(breeds))]
        
        bars = ax2.barh(breeds[::-1], probs[::-1], color=colors[::-1])
        ax2.set_xlabel("Confidence (%)")
        ax2.set_title("Top Predictions")
        ax2.set_xlim(0, 100)
        
        # Add percentage labels
        for bar, prob in zip(bars, probs[::-1]):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1f}%', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to: {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Predict dog breed from image")
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--model-name", type=str, default="resnet50",
                       help="Model architecture name")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of top predictions")
    parser.add_argument("--visualize", action="store_true",
                       help="Show visualization")
    parser.add_argument("--save-viz", type=str, default=None,
                       help="Path to save visualization")
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = DogBreedPredictor(
        model_path=args.model_path,
        model_name=args.model_name
    )
    
    # Predict
    predictions = predictor.predict(args.image, top_k=args.top_k)
    
    # Print results
    print("\n" + "="*50)
    print("DOG BREED PREDICTIONS")
    print("="*50)
    for i, (breed, prob) in enumerate(predictions, 1):
        print(f"{i}. {breed}: {prob*100:.2f}%")
    print("="*50)
    
    # Visualize if requested
    if args.visualize or args.save_viz:
        predictor.visualize_prediction(args.image, args.top_k, args.save_viz)


if __name__ == "__main__":
    main()
