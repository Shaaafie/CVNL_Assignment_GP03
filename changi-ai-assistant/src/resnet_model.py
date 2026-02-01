import torch
from torchvision import models
from pathlib import Path
from typing import List


def build_resnet_model(checkpoint_path: str, device='cpu'):
    """
    Load a ResNet model from checkpoint.
    Tries ResNet18, ResNet34, and ResNet50 variants.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded ResNet model
        class_names: List of class names
        img_size: Input image size
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    class_names: List[str] = ckpt.get('class_names', [])
    img_size = int(ckpt.get('img_size', 224))
    banner_px = int(ckpt.get('banner_px', 0))
    num_classes = len(class_names)
    
    # Try different ResNet architectures
    candidates = [
        models.resnet18,
        models.resnet34,
        models.resnet50,
    ]
    
    last_err = None
    for resnet_ctor in candidates:
        try:
            model = resnet_ctor(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(ckpt['model_state'], strict=True)
            model.to(device)
            model.eval()
            return model, class_names, img_size, banner_px
        except Exception as exc:
            last_err = exc
    
    raise RuntimeError(f"Failed to load checkpoint into ResNet variants: {last_err}")


def get_resnet_info(checkpoint_path: str):
    """Get model information without loading the full model"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return {
        'class_names': ckpt.get('class_names', []),
        'num_classes': len(ckpt.get('class_names', [])),
        'img_size': ckpt.get('img_size', 224),
        'banner_px': ckpt.get('banner_px', 0)
    }
