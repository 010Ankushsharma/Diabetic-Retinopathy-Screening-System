"""
Dataset classes for Diabetic Retinopathy detection.
Supports multiple datasets: Kaggle 2015, APTOS 2019, EyePACS, Messidor-2.
"""

import os
from typing import Optional, Tuple, List, Dict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class DRDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy classification.
    
    Args:
        image_dir: Path to directory containing images
        annotations: Path to CSV file with image labels
        transform: Albumentations transform to apply
        image_size: Target image size (height, width)
        is_test: Whether this is test mode (no labels)
    """
    
    DR_CLASSES = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }
    
    def __init__(
        self,
        image_dir: str,
        annotations: str,
        transform=None,
        image_size: int = 512,
        is_test: bool = False
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_size = image_size
        self.is_test = is_test
        
        # Load annotations
        self.df = pd.read_csv(annotations)
        
        # Standardize column names
        if 'id_code' in self.df.columns:
            self.image_col = 'id_code'
        elif 'image' in self.df.columns:
            self.image_col = 'image'
        else:
            raise ValueError("CSV must contain 'id_code' or 'image' column")
        
        if not is_test:
            if 'diagnosis' not in self.df.columns and 'level' not in self.df.columns:
                raise ValueError("CSV must contain 'diagnosis' or 'level' column")
            self.label_col = 'diagnosis' if 'diagnosis' in self.df.columns else 'level'
        
        print(f"Loaded {len(self.df)} images from {image_dir}")
        if not is_test:
            print(f"Class distribution:\n{self.df[self.label_col].value_counts().sort_index()}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        # Get image path
        img_name = self.df.iloc[idx][self.image_col]
        
        # Try different extensions
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            path = self.image_dir / f"{img_name}{ext}"
            if path.exists():
                img_path = path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get label if available
        if not self.is_test:
            label = int(self.df.iloc[idx][self.label_col])
        else:
            label = -1
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long) if label >= 0 else torch.tensor(-1),
            'image_name': img_name,
            'original_shape': image.shape
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling imbalance."""
        if self.is_test:
            return torch.ones(5)
        
        class_counts = self.df[self.label_col].value_counts().sort_index()
        total = len(self.df)
        num_classes = 5
        
        # Inverse frequency weighting
        weights = total / (num_classes * class_counts.reindex(range(num_classes), fill_value=1))
        return weights
    
    def get_sampler(self) -> WeightedRandomSampler:
        """Create weighted sampler for balanced training."""
        if self.is_test:
            return None
        
        class_counts = self.df[self.label_col].value_counts()
        samples_weight = [1.0 / class_counts[label] for label in self.df[self.label_col]]
        samples_weight = torch.DoubleTensor(samples_weight)
        
        return WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    
    @staticmethod
    def preprocess_ben_graham(image: np.ndarray, target_size: int = 512) -> np.ndarray:
        """
        Apply Ben Graham's preprocessing technique for retinal images.
        This helps normalize illumination and enhance features.
        """
        # Resize to manageable size for preprocessing
        original_shape = image.shape
        if image.shape[0] > 1000 or image.shape[1] > 1000:
            scale = 1000 / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge back
        limg = cv2.merge((cl, a, b))
        processed = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # Resize to target size
        processed = cv2.resize(processed, (target_size, target_size))
        
        return processed


class MultiDataset(Dataset):
    """
    Combines multiple datasets for training.
    Useful for domain adaptation and increasing dataset diversity.
    """
    
    def __init__(
        self,
        datasets: List[DRDataset],
        transform=None
    ):
        self.datasets = datasets
        self.transform = transform
        self.total_length = sum(len(d) for d in datasets)
        
        # Create cumulative length array for indexing
        self.cumulative_lengths = np.cumsum([0] + [len(d) for d in datasets])
        
        print(f"Combined dataset: {self.total_length} images from {len(datasets)} sources")
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Dict:
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        sample_idx = idx - self.cumulative_lengths[dataset_idx]
        
        return self.datasets[dataset_idx][sample_idx]


def create_dataloaders(
    train_dataset: DRDataset,
    val_dataset: DRDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_weighted_sampler: Whether to use weighted sampling for class balance
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Training loader with optional weighted sampling
    if use_weighted_sampler:
        sampler = train_dataset.get_sampler()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    # Validation loader (no shuffling needed)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
