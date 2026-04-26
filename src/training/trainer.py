"""
Training pipeline for DR classification model.
"""

import os
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from typing import Optional

from src.training.lit_model import DRClassifierModule
from src.data.dataset import create_dataloaders


def train(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: dict,
    checkpoint_dir: str = "models/checkpoints",
    wandb_project: str = "diabetic-retinopathy-screening",
    wandb_entity: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None
):
    """
    Train the DR classification model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        checkpoint_dir: Directory to save checkpoints
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize model
    model = DRClassifierModule(
        backbone=config['MODEL']['BACKBONE'],
        num_classes=config['MODEL']['NUM_CLASSES'],
        pretrained=config['MODEL']['PRETRAINED'],
        dropout=config['MODEL']['DROPOUT'],
        learning_rate=config['TRAINING']['LEARNING_RATE'],
        weight_decay=config['TRAINING']['WEIGHT_DECAY'],
        loss_type=config['TRAINING']['LOSS'],
        class_weights=torch.tensor(config['TRAINING']['CLASS_WEIGHTS']) if config['TRAINING'].get('CLASS_WEIGHTS') else None,
        focal_alpha=config['TRAINING'].get('FOCAL_ALPHA', 0.25),
        focal_gamma=config['TRAINING'].get('FOCAL_GAMMA', 2.0)
    )
    
    # Print model parameters
    param_counts = model.model.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Total: {param_counts['total_M']:.2f}M")
    print(f"  Trainable: {param_counts['trainable_M']:.2f}M")
    print(f"  Frozen: {param_counts['frozen']}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='dr-{epoch:02d}-{val_kappa:.4f}',
        save_top_k=3,
        monitor='val_kappa',
        mode='max',
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_kappa',
        patience=config['TRAINING']['PATIENCE'],
        mode='max',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    progress_bar = RichProgressBar()
    
    # Loggers
    loggers = []
    
    # W&B Logger
    wandb_logger = WandbLogger(
        project=wandb_project,
        entity=wandb_entity,
        log_model=True,
        config=config
    )
    loggers.append(wandb_logger)
    
    # TensorBoard Logger
    tb_logger = TensorBoardLogger(
        save_dir="logs/tensorboard",
        name=config['MODEL']['BACKBONE']
    )
    loggers.append(tb_logger)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['TRAINING']['EPOCHS'],
        accelerator='auto',
        devices='auto',
        precision='16-mixed' if config['TRAINING']['AMP'] else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor, progress_bar],
        logger=loggers,
        log_every_n_steps=config['LOGGING']['LOG_INTERVAL'],
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from_checkpoint
    )
    
    # Get best model path
    best_model_path = checkpoint_callback.best_model_path
    print(f"\nTraining completed!")
    print(f"Best model saved at: {best_model_path}")
    print(f"Best validation kappa: {checkpoint_callback.best_model_score:.4f}")
    
    return model, best_model_path


def train_with_config(config_path: str = "config.yaml"):
    """
    Train model using configuration file.
    
    Args:
        config_path: Path to YAML configuration file
    """
    import yaml
    from src.data.dataset import DRDataset
    from src.data.augmentation import get_train_transforms, get_val_transforms
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded:")
    print(f"  Backbone: {config['MODEL']['BACKBONE']}")
    print(f"  Image size: {config['MODEL']['IMAGE_SIZE']}")
    print(f"  Batch size: {config['TRAINING']['BATCH_SIZE']}")
    print(f"  Epochs: {config['TRAINING']['EPOCHS']}")
    print(f"  Learning rate: {config['TRAINING']['LEARNING_RATE']}")
    
    # Create datasets
    train_transform = get_train_transforms(
        image_size=config['MODEL']['IMAGE_SIZE'],
        brightness_limit=config['AUGMENTATION']['TRAIN']['RANDOM_BRIGHTNESS'],
        contrast_limit=config['AUGMENTATION']['TRAIN']['RANDOM_CONTRAST'],
        hflip_prob=config['AUGMENTATION']['TRAIN']['HFLIP'],
        vflip_prob=config['AUGMENTATION']['TRAIN']['VFLIP'],
        rotate_limit=config['AUGMENTATION']['TRAIN']['ROTATE'],
        blur_prob=config['AUGMENTATION']['TRAIN']['BLUR_PROB'],
        noise_prob=config['AUGMENTATION']['TRAIN']['NOISE_PROB'],
        use_clahe=config['AUGMENTATION']['TRAIN']['CLAHE']
    )
    
    val_transform = get_val_transforms(
        image_size=config['MODEL']['IMAGE_SIZE']
    )
    
    # Example: Load Kaggle 2015 dataset
    # You need to adjust paths based on your dataset structure
    train_dataset = DRDataset(
        image_dir=config['DATASETS']['KAGGLE_2015'] + '/train',
        annotations=config['DATASETS']['KAGGLE_2015'] + '/train_labels.csv',
        transform=train_transform,
        image_size=config['MODEL']['IMAGE_SIZE']
    )
    
    val_dataset = DRDataset(
        image_dir=config['DATASETS']['KAGGLE_2015'] + '/val',
        annotations=config['DATASETS']['KAGGLE_2015'] + '/val_labels.csv',
        transform=val_transform,
        image_size=config['MODEL']['IMAGE_SIZE']
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config['TRAINING']['BATCH_SIZE'],
        num_workers=config['TRAINING']['NUM_WORKERS'],
        use_weighted_sampler=True
    )
    
    # Train
    model, best_checkpoint = train(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=config['CHECKPOINT_DIR'],
        wandb_project=config['LOGGING']['WANDB_PROJECT'],
        wandb_entity=config['LOGGING']['WANDB_ENTITY']
    )
    
    return model, best_checkpoint


if __name__ == "__main__":
    # Run training with default config
    train_with_config("config.yaml")
