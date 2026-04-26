"""
Diabetic Retinopathy classification model using EfficientNet backbone.
"""

import torch
import torch.nn as nn
from typing import Optional
import timm


class DRClassifier(nn.Module):
    """
    EfficientNet-based classifier for Diabetic Retinopathy detection.
    
    Architecture:
    - Backbone: EfficientNet-B4 (or other variants)
    - Global Average Pooling
    - Dropout
    - Fully connected layer for 5-class classification
    
    Args:
        backbone_name: Name of the backbone model (default: efficientnet_b4)
        num_classes: Number of output classes (default: 5)
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super(DRClassifier, self).__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Load backbone model
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            global_pool='avg'
        )
        
        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
        
        print(f"Model: {backbone_name}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Number of classes: {num_classes}")
    
    def _init_classifier(self):
        """Initialize classifier layer weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.
        Useful for Grad-CAM and feature visualization.
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor
        """
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")
    
    def freeze_backbone_except_last_n(self, n: int = 2):
        """
        Freeze all backbone except last n blocks.
        
        Args:
            n: Number of blocks to keep trainable
        """
        # Freeze all first
        self.freeze_backbone()
        
        # Get backbone children
        children = list(self.backbone.children())
        
        # Unfreeze last n blocks
        for child in children[-n:]:
            for param in child.parameters():
                param.requires_grad = True
        
        print(f"Backbone frozen except last {n} blocks")
    
    def count_parameters(self) -> dict:
        """
        Count trainable and total parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
            'total_M': total_params / 1e6,
            'trainable_M': trainable_params / 1e6
        }
    
    @staticmethod
    def get_model_variants():
        """Get available model variants."""
        return {
            'lightweight': 'efficientnet_b0',  # For mobile
            'standard': 'efficientnet_b4',     # For server
            'high_accuracy': 'efficientnet_b7' # For maximum accuracy
        }


def create_model(
    backbone: str = "efficientnet_b4",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.3,
    device: str = "cuda"
) -> DRClassifier:
    """
    Factory function to create DR classifier model.
    
    Args:
        backbone: Backbone architecture name
        num_classes: Number of classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
        device: Device to create model on
    
    Returns:
        DRClassifier model
    """
    model = DRClassifier(
        backbone_name=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    
    model = model.to(device)
    
    return model
