"""
Data Loading Utilities for Dog Breed Classification
=====================================================
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image


# Standard ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224, is_training: bool = True):
    """
    Get image transforms for training or evaluation.
    
    Args:
        image_size: Target image size (default: 224 for most pretrained models)
        is_training: Whether to include data augmentation
    
    Returns:
        torchvision.transforms.Compose object
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # 256 for 224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory containing train/, val/, test/ folders
        batch_size: Batch size for training
        image_size: Target image size
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    data_dir = Path(data_dir)
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        data_dir / "train",
        transform=get_transforms(image_size, is_training=True)
    )
    
    val_dataset = datasets.ImageFolder(
        data_dir / "val",
        transform=get_transforms(image_size, is_training=False)
    )
    
    test_dataset = datasets.ImageFolder(
        data_dir / "test",
        transform=get_transforms(image_size, is_training=False)
    )
    
    # Get class names
    class_names = train_dataset.classes
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    print(f"  Classes: {len(class_names)} breeds")
    
    return train_loader, val_loader, test_loader, class_names


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize an image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor
    
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


if __name__ == "__main__":
    # Quick test
    data_dir = Path(__file__).parent.parent / "data"
    if (data_dir / "train").exists():
        train_loader, val_loader, test_loader, classes = get_dataloaders(str(data_dir))
        print(f"\nSample batch shape: {next(iter(train_loader))[0].shape}")
    else:
        print("Data not found. Run download_dataset.py first.")
