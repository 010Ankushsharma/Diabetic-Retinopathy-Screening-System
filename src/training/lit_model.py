"""
PyTorch Lightning module for DR classification.
Handles training, validation, and logging.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing import Optional

from src.models.dr_model import DRClassifier
from src.models.losses import get_loss_function
from src.models.metrics import QuadraticWeightedKappa, compute_metrics


class DRClassifierModule(pl.LightningModule):
    """
    PyTorch Lightning module for Diabetic Retinopathy classification.
    
    Args:
        backbone: Backbone architecture name
        num_classes: Number of output classes
        pretrained: Use pretrained backbone
        dropout: Dropout rate
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        loss_type: Loss function type ('focal' or 'weighted_ce')
        class_weights: Class weights tensor
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        loss_type: str = "focal",
        class_weights: Optional[torch.Tensor] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = DRClassifier(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
        
        # Loss function
        self.loss_fn = get_loss_function(
            loss_type=loss_type,
            class_weights=class_weights,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )
        
        # Metrics
        self.train_kappa = QuadraticWeightedKappa(num_classes)
        self.val_kappa = QuadraticWeightedKappa(num_classes)
        
        # Store predictions and targets for epoch-end metrics
        self.val_predictions = []
        self.val_targets = []
        self.train_predictions = []
        self.train_targets = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        outputs = self(images)
        
        # Compute loss
        loss = self.loss_fn(outputs, labels)
        
        # Get predictions
        predictions = torch.argmax(outputs, dim=1)
        
        # Update metrics
        self.train_kappa.update(predictions, labels)
        self.train_predictions.extend(predictions.cpu().numpy().tolist())
        self.train_targets.extend(labels.cpu().numpy().tolist())
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Compute and log epoch-end training metrics."""
        kappa = self.train_kappa.compute()
        self.log('train_kappa', kappa, on_epoch=True, prog_bar=True, logger=True)
        
        # Log to W&B
        if self.logger is not None:
            self.logger.experiment.log({'train_kappa_epoch': kappa})
        
        self.train_kappa.reset()
        self.train_predictions = []
        self.train_targets = []
    
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step."""
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        outputs = self(images)
        
        # Compute loss
        loss = self.loss_fn(outputs, labels)
        
        # Get predictions
        predictions = torch.argmax(outputs, dim=1)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Update metrics
        self.val_kappa.update(predictions, labels)
        self.val_predictions.extend(predictions.cpu().numpy().tolist())
        self.val_targets.extend(labels.cpu().numpy().tolist())
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        
        return {'val_loss': loss, 'predictions': predictions, 'targets': labels}
    
    def on_validation_epoch_end(self):
        """Compute and log epoch-end validation metrics."""
        kappa = self.val_kappa.compute()
        self.log('val_kappa', kappa, on_epoch=True, prog_bar=True, logger=True)
        
        # Compute comprehensive metrics
        import numpy as np
        predictions_array = np.array(self.val_predictions)
        targets_array = np.array(self.val_targets)
        
        metrics = compute_metrics(predictions_array, targets_array)
        
        # Log all metrics
        self.log('val_accuracy', metrics['accuracy'], on_epoch=True, logger=True)
        self.log('val_f1_macro', metrics['f1_macro'], on_epoch=True, logger=True)
        self.log('val_kappa_final', metrics['quadratic_kappa'], on_epoch=True, logger=True)
        
        # Log to W&B
        if self.logger is not None:
            self.logger.experiment.log({
                'val_accuracy_epoch': metrics['accuracy'],
                'val_f1_macro_epoch': metrics['f1_macro'],
                'val_kappa_epoch': metrics['quadratic_kappa'],
                'val_sensitivity': metrics['sensitivity_referable'],
                'val_specificity': metrics['specificity_referable']
            })
        
        # Print metrics every 10 epochs
        if self.current_epoch % 10 == 0:
            from src.models.metrics import print_metrics
            print_metrics(metrics)
        
        self.val_kappa.reset()
        self.val_predictions = []
        self.val_targets = []
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
                div_factor=25,
                final_div_factor=1000
            ),
            'interval': 'step',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]
    
    def get_predictions(self, images: torch.Tensor) -> dict:
        """
        Get predictions for inference.
        
        Args:
            images: Input images tensor
        
        Returns:
            Dictionary with predictions and probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'logits': outputs
        }
