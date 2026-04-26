"""
Example training script for DR classification.
This script demonstrates the complete training workflow.
"""

import yaml
import torch
from pathlib import Path

from src.data.dataset import DRDataset, create_dataloaders
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.training.trainer import train


def main():
    """Main training function."""
    print("="*80)
    print("DIABETIC RETINOPATHY CLASSIFICATION - TRAINING PIPELINE")
    print("="*80)
    
    # Load configuration
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nConfiguration loaded:")
    print(f"  Backbone: {config['MODEL']['BACKBONE']}")
    print(f"  Image Size: {config['MODEL']['IMAGE_SIZE']}")
    print(f"  Batch Size: {config['TRAINING']['BATCH_SIZE']}")
    print(f"  Epochs: {config['TRAINING']['EPOCHS']}")
    print(f"  Loss Function: {config['TRAINING']['LOSS']}")
    print(f"  Learning Rate: {config['TRAINING']['LEARNING_RATE']}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f} GB")
    else:
        print("\n⚠ No GPU detected. Training on CPU (will be slow)")
    
    # Create transforms
    print("\nCreating data augmentation pipelines...")
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
    
    # Create datasets
    # NOTE: Adjust paths based on your dataset structure
    train_dataset_path = config['DATASETS']['KAGGLE_2015']
    val_dataset_path = config['DATASETS']['APTOS_2019']
    
    print(f"\nLoading training dataset from: {train_dataset_path}")
    train_dataset = DRDataset(
        image_dir=f"{train_dataset_path}/train",
        annotations=f"{train_dataset_path}/train_labels.csv",
        transform=train_transform,
        image_size=config['MODEL']['IMAGE_SIZE']
    )
    
    print(f"\nLoading validation dataset from: {val_dataset_path}")
    val_dataset = DRDataset(
        image_dir=f"{val_dataset_path}/images",
        annotations=f"{val_dataset_path}/train.csv",
        transform=val_transform,
        image_size=config['MODEL']['IMAGE_SIZE']
    )
    
    # Create dataloaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config['TRAINING']['BATCH_SIZE'],
        num_workers=config['TRAINING']['NUM_WORKERS'],
        use_weighted_sampler=True
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Create checkpoint directory
    checkpoint_dir = config['CHECKPOINT_DIR']
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Train model
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    model, best_checkpoint = train(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=checkpoint_dir,
        wandb_project=config['LOGGING']['WANDB_PROJECT'],
        wandb_entity=config['LOGGING']['WANDB_ENTITY']
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best model saved at: {best_checkpoint}")
    
    # Export to ONNX
    print("\nExporting model to ONNX format...")
    from src.inference.export import export_from_config
    
    onnx_path, optimized_path, mobile_path = export_from_config(
        checkpoint_path=best_checkpoint,
        config_path=config_path,
        output_dir=config['ONNX_DIR']
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"PyTorch checkpoint: {best_checkpoint}")
    print(f"ONNX model: {onnx_path}")
    print(f"Optimized ONNX: {optimized_path}")
    print(f"Mobile ONNX: {mobile_path}")
    
    print("\nNext steps:")
    print("1. Test the model: python src/inference/inference.py --image test_image.jpg")
    print("2. Start API server: uvicorn src.api.main:app --reload")
    print("3. Deploy with Docker: docker-compose up -d")


if __name__ == "__main__":
    main()
