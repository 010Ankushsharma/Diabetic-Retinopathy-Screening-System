"""
Grad-CAM implementation for visualizing regions of interest in retinal images.
Highlights lesions: microaneurysms, hemorrhages, and exudates.
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path


class GradCAMExplainer:
    """
    Grad-CAM explainer for DR classification models.
    
    Generates heatmaps showing which regions of the retinal image
    contributed most to the model's prediction.
    
    Args:
        model: Trained DR classifier model
        target_layer: Target convolutional layer for Grad-CAM
        method: Grad-CAM method ('gradcam', 'gradcam++', 'eigencam')
        use_cuda: Whether to use GPU
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: str = "features.9",
        method: str = "gradcam",
        use_cuda: bool = True
    ):
        self.model = model
        self.target_layer_name = target_layer
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Get the target layer
        self.target_layer = self._get_target_layer(target_layer)
        
        # Initialize Grad-CAM
        if method == "gradcam":
            self.cam = GradCAM(model=self.model, target_layers=[self.target_layer], use_cuda=self.use_cuda)
        elif method == "gradcam++":
            self.cam = GradCAMPlusPlus(model=self.model, target_layers=[self.target_layer], use_cuda=self.use_cuda)
        elif method == "eigencam":
            self.cam = EigenCAM(model=self.model, target_layers=[self.target_layer], use_cuda=self.use_cuda)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.method = method
    
    def _get_target_layer(self, layer_name: str) -> torch.nn.Module:
        """
        Get target layer from model.
        
        Args:
            layer_name: Name or path to the layer
        
        Returns:
            Target layer module
        """
        # For EfficientNet, typically the last conv layer
        if hasattr(self.model, 'model'):
            # PyTorch Lightning wrapper
            backbone = self.model.model.backbone
        else:
            backbone = self.model.backbone
        
        # Try to get layer by name
        try:
            target = backbone
            for part in layer_name.split('.'):
                if part.isdigit():
                    target = target[int(part)]
                else:
                    target = getattr(target, part)
            return target
        except:
            # Fallback: use last layer
            print(f"Warning: Could not find layer {layer_name}, using last layer")
            return list(backbone.children())[-1]
    
    def generate_heatmap(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an image.
        
        Args:
            image: Input image tensor (1, 3, H, W)
            target_class: Target class for Grad-CAM (None for predicted class)
        
        Returns:
            Heatmap array (H, W) normalized to [0, 1]
        """
        # Move to device
        if self.use_cuda:
            image = image.cuda()
        
        # Get predicted class if not provided
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = torch.argmax(output, dim=1).item()
        
        # Generate CAM
        grayscale_cam = self.cam(
            input_tensor=image,
            targets=[target_class],
            eigen_smooth=True
        )
        
        # Get the first (and only) image's CAM
        heatmap = grayscale_cam[0, :]
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original RGB image (H, W, 3)
            heatmap: Heatmap array (H, W) in [0, 1]
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap
        
        Returns:
            Image with heatmap overlay (H, W, 3)
        """
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image_normalized = image.astype(np.float32) / 255.0
        else:
            image_normalized = image
        
        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            colormap
        )
        heatmap_colored = heatmap_colored.astype(np.float32) / 255.0
        
        # Blend
        overlay = cv2.addWeighted(
            image_normalized,
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )
        
        # Clip values
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
    
    def explain(
        self,
        image: torch.Tensor,
        original_image: Optional[np.ndarray] = None,
        target_class: Optional[int] = None
    ) -> dict:
        """
        Generate complete explanation with heatmap and overlay.
        
        Args:
            image: Input image tensor (1, 3, H, W)
            original_image: Original RGB image for overlay (H, W, 3)
            target_class: Target class (None for predicted class)
        
        Returns:
            Dictionary with heatmap and overlay
        """
        # Generate heatmap
        heatmap = self.generate_heatmap(image, target_class)
        
        # Create overlay if original image provided
        overlay = None
        if original_image is not None:
            overlay = self.overlay_heatmap(original_image, heatmap)
        
        return {
            'heatmap': heatmap,
            'overlay': overlay,
            'target_class': target_class
        }
    
    def batch_explain(
        self,
        images: torch.Tensor,
        original_images: Optional[np.ndarray] = None,
        target_classes: Optional[list] = None
    ) -> list:
        """
        Generate explanations for a batch of images.
        
        Args:
            images: Batch of image tensors (B, 3, H, W)
            original_images: Batch of original RGB images (B, H, W, 3)
            target_classes: List of target classes
        
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for i in range(images.shape[0]):
            image = images[i:i+1]
            original = original_images[i] if original_images is not None else None
            target = target_classes[i] if target_classes is not None else None
            
            explanation = self.explain(image, original, target)
            explanations.append(explanation)
        
        return explanations


def generate_heatmap(
    model: torch.nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    target_class: Optional[int] = None,
    target_layer: str = "features.9",
    method: str = "gradcam"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to generate Grad-CAM heatmap and overlay.
    
    Args:
        model: Trained model
        image: Input image tensor (1, 3, H, W)
        original_image: Original RGB image (H, W, 3)
        target_class: Target class (None for predicted)
        target_layer: Target layer name
        method: Grad-CAM method
    
    Returns:
        Tuple of (heatmap, overlay)
    """
    explainer = GradCAMExplainer(
        model=model,
        target_layer=target_layer,
        method=method
    )
    
    result = explainer.explain(image, original_image, target_class)
    
    return result['heatmap'], result['overlay']


def validate_heatmap_against_mask(
    heatmap: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Validate Grad-CAM heatmap against ground truth lesion masks.
    Useful for IDRiD dataset with pixel-level annotations.
    
    Args:
        heatmap: Grad-CAM heatmap (H, W) in [0, 1]
        ground_truth_mask: Ground truth binary mask (H, W)
        threshold: Threshold for binarizing heatmap
    
    Returns:
        Dictionary with validation metrics
    """
    # Binarize heatmap
    binary_heatmap = (heatmap > threshold).astype(np.float32)
    
    # Ensure same shape
    if binary_heatmap.shape != ground_truth_mask.shape:
        binary_heatmap = cv2.resize(
            binary_heatmap,
            (ground_truth_mask.shape[1], ground_truth_mask.shape[0])
        )
    
    # Compute IoU (Intersection over Union)
    intersection = np.sum(binary_heatmap * ground_truth_mask)
    union = np.sum(np.clip(binary_heatmap + ground_truth_mask, 0, 1))
    iou = intersection / (union + 1e-8)
    
    # Compute overlap percentage
    lesion_area = np.sum(ground_truth_mask)
    overlap_area = intersection
    overlap_percentage = overlap_area / (lesion_area + 1e-8) * 100
    
    return {
        'iou': iou,
        'overlap_percentage': overlap_percentage,
        'heatmap_area': np.sum(binary_heatmap),
        'lesion_area': lesion_area
    }
