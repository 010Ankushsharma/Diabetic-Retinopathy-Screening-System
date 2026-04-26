"""
Image preprocessing and quality assessment for retinal images.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class QualityChecker:
    """
    Assesses image quality to reject blurry or poor-quality images.
    Critical for medical AI systems to avoid false predictions.
    """
    
    def __init__(
        self,
        min_sharpness: float = 100.0,
        min_brightness: int = 20,
        max_brightness: int = 230
    ):
        self.min_sharpness = min_sharpness
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
    
    def check_quality(self, image: np.ndarray) -> Tuple[bool, dict]:
        """
        Check if image meets quality standards.
        
        Args:
            image: Input RGB image (numpy array)
        
        Returns:
            Tuple of (is_acceptable, quality_metrics)
        """
        metrics = {}
        
        # 1. Sharpness check (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = sharpness
        
        # 2. Brightness check
        brightness = np.mean(gray)
        metrics['brightness'] = brightness
        
        # 3. Contrast check
        contrast = np.std(gray)
        metrics['contrast'] = contrast
        
        # 4. Check for uniform/blank images
        entropy = self._calculate_entropy(gray)
        metrics['entropy'] = entropy
        
        # Quality decisions
        is_acceptable = True
        reasons = []
        
        if sharpness < self.min_sharpness:
            is_acceptable = False
            reasons.append(f"Image too blurry (sharpness: {sharpness:.2f} < {self.min_sharpness})")
        
        if brightness < self.min_brightness:
            is_acceptable = False
            reasons.append(f"Image too dark (brightness: {brightness:.2f} < {self.min_brightness})")
        
        if brightness > self.max_brightness:
            is_acceptable = False
            reasons.append(f"Image too bright (brightness: {brightness:.2f} > {self.max_brightness})")
        
        if entropy < 3.0:
            is_acceptable = False
            reasons.append("Image appears uniform/blank")
        
        metrics['is_acceptable'] = is_acceptable
        metrics['reasons'] = reasons
        
        return is_acceptable, metrics
    
    @staticmethod
    def _calculate_entropy(gray_image: np.ndarray) -> float:
        """Calculate image entropy."""
        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        histogram = histogram.flatten() / histogram.sum()
        
        # Remove zeros to avoid log(0)
        histogram = histogram[histogram > 0]
        entropy = -np.sum(histogram * np.log2(histogram))
        
        return entropy


class Preprocessor:
    """
    Preprocessing pipeline for retinal fundus images.
    Implements Ben Graham's technique and other enhancements.
    """
    
    def __init__(
        self,
        image_size: int = 512,
        use_clahe: bool = True,
        remove_black_borders: bool = True
    ):
        self.image_size = image_size
        self.use_clahe = use_clahe
        self.remove_black_borders = remove_black_border
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline.
        
        Args:
            image: Input RGB image
        
        Returns:
            Preprocessed image
        """
        # Step 1: Remove black borders
        if self.remove_black_borders:
            image = self._remove_black_borders(image)
        
        # Step 2: Resize to manageable size for processing
        original_shape = image.shape
        if image.shape[0] > 1000 or image.shape[1] > 1000:
            scale = 1000 / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Step 3: Apply Ben Graham's preprocessing
        image = self._ben_graham_preprocessing(image)
        
        # Step 4: Resize to target size
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        return image
    
    def _remove_black_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove black borders from retinal images."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Crop image
            image = image[y:y+h, x:x+w]
        
        return image
    
    def _ben_graham_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Ben Graham's preprocessing technique.
        Normalizes illumination and enhances blood vessels.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel if enabled
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
        
        # Merge channels back
        processed = cv2.merge((l, a, b))
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)
        
        return processed
    
    def preprocess_batch(self, images: list) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of numpy arrays
        
        Returns:
            Stack of preprocessed images
        """
        processed_images = [self.preprocess(img) for img in images]
        return np.stack(processed_images)
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def standardize_image(
        image: np.ndarray,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> np.ndarray:
        """
        Standardize image using ImageNet statistics.
        
        Args:
            image: Input image in [0, 1] range
            mean: Channel means
            std: Channel standard deviations
        
        Returns:
            Standardized image
        """
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        
        return (image - mean) / std
