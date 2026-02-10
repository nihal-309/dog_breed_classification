"""
Model Definitions for Dog Breed Classification
================================================
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


def get_model(
    model_name: str = "resnet50",
    num_classes: int = 120,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Get a pretrained model for dog breed classification.
    
    Args:
        model_name: Name of the model architecture
            Options: 'resnet50', 'resnet101', 'efficientnet_b0', 'efficientnet_b3',
                     'mobilenet_v2', 'vgg16', 'densenet121'
        num_classes: Number of output classes (120 for Stanford Dogs)
        pretrained: Whether to use pretrained ImageNet weights
        freeze_backbone: Whether to freeze backbone layers
    
    Returns:
        PyTorch model ready for training
    """
    
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "resnet101":
        weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet101(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(4096, num_classes)
        
    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Choose from: resnet50, resnet101, efficientnet_b0, "
                        f"efficientnet_b3, mobilenet_v2, vgg16, densenet121")
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the classifier
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                for param in model.classifier[-1].parameters():
                    param.requires_grad = True
            else:
                for param in model.classifier.parameters():
                    param.requires_grad = True
    
    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with total, trainable, and frozen parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6
    }


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    for name in ["resnet50", "efficientnet_b0", "mobilenet_v2"]:
        model = get_model(name, num_classes=120, pretrained=False)
        params = count_parameters(model)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        
        print(f"\n{name}:")
        print(f"  Parameters: {params['total_millions']:.2f}M")
        print(f"  Output shape: {y.shape}")
