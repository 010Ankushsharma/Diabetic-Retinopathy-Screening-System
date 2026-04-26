"""
Custom loss functions for handling class imbalance in DR detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focuses training on hard examples by down-weighting easy examples.
    Particularly effective for highly imbalanced datasets like DR.
    
    Args:
        alpha: Weighting factor for each class (can be tensor or float)
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        
        Returns:
            Focal loss value
        """
        # Get probabilities using softmax
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropy(nn.Module):
    """
    Weighted Cross Entropy Loss for class imbalance.
    
    Args:
        class_weights: Weight for each class
        reduction: Reduction method
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super(WeightedCrossEntropy, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross entropy.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth labels
        
        Returns:
            Weighted cross entropy loss
        """
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.class_weights,
            reduction=self.reduction
        )


def get_loss_function(
    loss_type: str = "focal",
    class_weights: Optional[torch.Tensor] = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0
) -> nn.Module:
    """
    Factory function to create loss function.
    
    Args:
        loss_type: Type of loss ('focal' or 'weighted_ce')
        class_weights: Class weights tensor
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
    
    Returns:
        Loss function
    """
    if loss_type == "focal":
        if class_weights is not None:
            alpha = class_weights / class_weights.sum()
        else:
            alpha = focal_alpha
        
        return FocalLoss(alpha=alpha, gamma=focal_gamma, reduction='mean')
    
    elif loss_type == "weighted_ce":
        return WeightedCrossEntropy(class_weights=class_weights, reduction='mean')
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
