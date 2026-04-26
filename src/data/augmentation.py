"""
Data augmentation pipelines for training and validation.
Uses Albumentations for efficient augmentation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional


def get_train_transforms(
    image_size: int = 512,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
    rotate_limit: int = 360,
    blur_prob: float = 0.1,
    noise_prob: float = 0.1,
    use_clahe: bool = True
) -> A.Compose:
    """
    Create training augmentation pipeline.
    
    Args:
        image_size: Target image size
        brightness_limit: Brightness augmentation range
        contrast_limit: Contrast augmentation range
        hflip_prob: Horizontal flip probability
        vflip_prob: Vertical flip probability
        rotate_limit: Maximum rotation angle
        blur_prob: Probability of applying blur
        noise_prob: Probability of adding noise
        use_clahe: Whether to apply CLAHE
    
    Returns:
        Albumentations Compose transform
    """
    transforms = []
    
    # Resize
    transforms.append(A.Resize(image_size, image_size))
    
    # CLAHE for contrast enhancement
    if use_clahe:
        transforms.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5))
    
    # Color augmentations
    transforms.append(
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.8
        )
    )
    
    # Hue and saturation
    transforms.append(
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.5
        )
    )
    
    # Geometric augmentations
    transforms.append(A.HorizontalFlip(p=hflip_prob))
    transforms.append(A.VerticalFlip(p=vflip_prob))
    transforms.append(A.Rotate(limit=rotate_limit, p=0.8))
    
    # Shift and scale
    transforms.append(
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5
        )
    )
    
    # Simulate real-world conditions
    if blur_prob > 0:
        transforms.append(
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ], p=blur_prob)
        )
    
    if noise_prob > 0:
        transforms.append(
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=noise_prob)
        )
    
    # Grid distortion for realistic变形
    transforms.append(A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3))
    
    # Optical distortion
    transforms.append(
        A.OpticalDistortion(
            distort_limit=0.2,
            shift_limit=0.05,
            p=0.3
        )
    )
    
    # Cutout for regularization
    transforms.append(
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=4,
            min_height=16,
            min_width=16,
            p=0.3
        )
    )
    
    # Normalization
    transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    # Convert to PyTorch tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_val_transforms(
    image_size: int = 512,
    center_crop: bool = True
) -> A.Compose:
    """
    Create validation/test augmentation pipeline.
    Minimal augmentation for validation.
    
    Args:
        image_size: Target image size
        center_crop: Whether to apply center crop
    
    Returns:
        Albumentations Compose transform
    """
    transforms = []
    
    # Resize with optional center crop
    if center_crop:
        transforms.append(A.LongestMaxSize(max_size=image_size))
        transforms.append(A.CenterCrop(image_size, image_size))
    else:
        transforms.append(A.Resize(image_size, image_size))
    
    # Normalization
    transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    # Convert to PyTorch tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_test_time_augmentation(
    image_size: int = 512,
    num_augments: int = 5
) -> A.Compose:
    """
    Create test-time augmentation pipeline.
    Multiple views of the same image for better predictions.
    
    Args:
        image_size: Target image size
        num_augments: Number of augmentations to generate
    
    Returns:
        Albumentations Compose transform
    """
    transforms = []
    
    transforms.append(A.Resize(image_size, image_size))
    
    # Test-time augmentations
    transforms.append(
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Transpose(p=1.0),
            A.NoOp(p=1.0),
        ], p=1.0)
    )
    
    # Normalization
    transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    # Convert to PyTorch tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)
