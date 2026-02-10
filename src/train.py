"""
Training Script for Dog Breed Classification
==============================================
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import get_dataloaders
from src.model import get_model, count_parameters


class Trainer:
    """Trainer class for dog breed classification."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=30,  # Will be updated based on epochs
            eta_min=1e-6
        )
        
        # Tracking
        self.best_val_acc = 0.0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
    
    def train_epoch(self) -> tuple:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.*correct/total:.2f}%"
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.val_loader, desc="Validating", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs: int, save_dir: str = "models"):
        """
        Full training loop.
        
        Args:
            epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nTraining for {epochs} epochs...")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_acc": val_acc,
                    "history": self.history
                }, save_dir / "best_model.pth")
                print(f"  → Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Save final model
        torch.save({
            "epoch": epochs,
            "model_state_dict": self.model.state_dict(),
            "history": self.history
        }, save_dir / "final_model.pth")
        
        print("-" * 60)
        print(f"Training complete! Best Val Acc: {self.best_val_acc:.2f}%")
        
        return self.history


def main():
    parser = argparse.ArgumentParser(description="Train Dog Breed Classifier")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--model", type=str, default="resnet50",
                       choices=["resnet50", "resnet101", "efficientnet_b0", 
                               "efficientnet_b3", "mobilenet_v2", "vgg16", "densenet121"],
                       help="Model architecture")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--image-size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--freeze-backbone", action="store_true",
                       help="Freeze backbone layers")
    
    args = parser.parse_args()
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = get_model(
        args.model,
        num_classes=len(class_names),
        pretrained=True,
        freeze_backbone=args.freeze_backbone
    )
    
    params = count_parameters(model)
    print(f"Total parameters: {params['total_millions']:.2f}M")
    print(f"Trainable parameters: {params['trainable_millions']:.2f}M")
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr
    )
    
    history = trainer.train(
        epochs=args.epochs,
        save_dir=f"models/{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


if __name__ == "__main__":
    main()
