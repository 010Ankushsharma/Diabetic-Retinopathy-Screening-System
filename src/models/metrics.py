"""
Evaluation metrics for DR classification.
Includes Quadratic Weighted Kappa (critical for DR), F1-score, and Accuracy.
"""

import torch
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple, Optional


class QuadraticWeightedKappa:
    """
    Quadratic Weighted Kappa metric.
    This is THE most important metric for DR detection challenges.
    
    Kappa measures agreement between raters, accounting for chance agreement.
    Quadratic weighting penalizes larger disagreements more heavily.
    """
    
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update predictions and targets.
        
        Args:
            predictions: Predicted class indices or logits
            targets: Ground truth class indices
        """
        # Convert logits to class indices if needed
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=1)
        
        self.predictions.extend(predictions.cpu().numpy().tolist())
        self.targets.extend(targets.cpu().numpy().tolist())
    
    def compute(self) -> float:
        """
        Compute quadratic weighted kappa.
        
        Returns:
            Kappa score (-1 to 1, where 1 is perfect agreement)
        """
        if len(self.predictions) == 0:
            return 0.0
        
        kappa = cohen_kappa_score(
            self.targets,
            self.predictions,
            weights='quadratic',
            labels=list(range(self.num_classes))
        )
        
        return kappa
    
    def reset(self):
        """Reset stored predictions and targets."""
        self.predictions = []
        self.targets = []


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 5
) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted class labels
        targets: Ground truth labels
        num_classes: Number of classes
    
    Returns:
        Dictionary containing all metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(targets, predictions)
    
    # Quadratic weighted kappa (most important for DR)
    kappa = cohen_kappa_score(
        targets,
        predictions,
        weights='quadratic',
        labels=list(range(num_classes))
    )
    
    # F1 scores
    f1_macro = f1_score(targets, predictions, average='macro')
    f1_micro = f1_score(targets, predictions, average='micro')
    
    # Per-class F1 scores
    f1_per_class = f1_score(targets, predictions, average=None, labels=list(range(num_classes)))
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
    
    # Classification report
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    report = classification_report(
        targets,
        predictions,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True
    )
    
    # Sensitivity and Specificity for referable DR (classes 3, 4)
    binary_targets = (targets >= 3).astype(int)
    binary_predictions = (predictions >= 3).astype(int)
    
    tp = np.sum((binary_predictions == 1) & (binary_targets == 1))
    tn = np.sum((binary_predictions == 0) & (binary_targets == 0))
    fp = np.sum((binary_predictions == 1) & (binary_targets == 0))
    fn = np.sum((binary_predictions == 0) & (binary_targets == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'quadratic_kappa': kappa,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'sensitivity_referable': sensitivity,
        'specificity_referable': specificity,
        'classification_report': report
    }
    
    return metrics


def print_metrics(metrics: Dict):
    """
    Print metrics in a readable format.
    
    Args:
        metrics: Metrics dictionary from compute_metrics
    """
    print("\n" + "="*60)
    print("DIABETIC RETINOPATHY CLASSIFICATION METRICS")
    print("="*60)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Quadratic Weighted Kappa: {metrics['quadratic_kappa']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Micro): {metrics['f1_micro']:.4f}")
    
    print(f"\nReferable DR Detection (Classes 3-4):")
    print(f"  Sensitivity: {metrics['sensitivity_referable']:.4f}")
    print(f"  Specificity: {metrics['specificity_referable']:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    for i, name in enumerate(class_names):
        print(f"  {name:20s}: {metrics['f1_per_class'][i]:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    header = "Predicted: " + "  ".join([f"{i:>8d}" for i in range(5)])
    print(header)
    for i, row in enumerate(cm):
        print(f"Actual {i}: " + "  ".join([f"{val:>8d}" for val in row]))
    
    print("="*60)
