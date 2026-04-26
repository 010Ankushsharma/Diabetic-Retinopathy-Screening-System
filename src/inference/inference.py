"""
Inference pipeline for DR classification.
Supports both PyTorch and ONNX models.
"""

import os
import cv2
import numpy as np
import torch
from typing import Optional, Dict, Tuple
from pathlib import Path

from src.models.dr_model import DRClassifier
from src.data.preprocessing import Preprocessor, QualityChecker
from src.explainability.gradcam import GradCAMExplainer


class DRInference:
    """
    Production inference engine for DR classification.
    
    Args:
        model_path: Path to model checkpoint or ONNX file
        config: Configuration dictionary
        device: Device to run inference on
        use_onnx: Whether to use ONNX runtime
    """
    
    DR_LABELS = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }
    
    def __init__(
        self,
        model_path: str,
        config: dict,
        device: str = "cuda",
        use_onnx: bool = False
    ):
        self.config = config
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_onnx = use_onnx
        
        # Initialize preprocessor
        self.preprocessor = Preprocessor(
            image_size=config['MODEL']['IMAGE_SIZE'],
            use_clahe=config['AUGMENTATION']['TRAIN']['CLAHE']
        )
        
        # Initialize quality checker
        self.quality_checker = QualityChecker(
            min_sharpness=config['QUALITY_CHECK']['MIN_SHARPNESS'],
            min_brightness=config['QUALITY_CHECK']['MIN_BRIGHTNESS'],
            max_brightness=config['QUALITY_CHECK']['MAX_BRIGHTNESS']
        )
        
        # Load model
        if use_onnx:
            self.onnx_session = self._load_onnx_model(model_path)
            self.model = None
        else:
            self.model = self._load_pytorch_model(model_path)
            self.onnx_session = None
            
            # Initialize Grad-CAM
            self.gradcam = GradCAMExplainer(
                model=self.model,
                target_layer=config['GRADCAM']['TARGET_LAYER'],
                method=config['GRADCAM']['METHOD'],
                use_cuda=self.device == "cuda"
            )
    
    def _load_pytorch_model(self, model_path: str) -> DRClassifier:
        """Load PyTorch model from checkpoint."""
        print(f"Loading PyTorch model from {model_path}")
        
        model = DRClassifier(
            backbone_name=self.config['MODEL']['BACKBONE'],
            num_classes=self.config['MODEL']['NUM_CLASSES'],
            pretrained=False,
            dropout=self.config['MODEL']['DROPOUT']
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle PyTorch Lightning checkpoints
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        print("Model loaded successfully")
        return model
    
    def _load_onnx_model(self, model_path: str):
        """Load ONNX model."""
        import onnxruntime as ort
        
        print(f"Loading ONNX model from {model_path}")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        
        print("ONNX model loaded successfully")
        return session
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load and preprocess image for inference.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (tensor, original_image)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Check quality
        if self.config['QUALITY_CHECK']['ENABLED']:
            is_acceptable, quality_metrics = self.quality_checker.check_quality(image_rgb)
            if not is_acceptable:
                print(f"Warning: Image quality issues detected:")
                for reason in quality_metrics['reasons']:
                    print(f"  - {reason}")
        
        # Preprocess
        processed = self.preprocessor.preprocess(image_rgb)
        
        # Normalize
        normalized = self.preprocessor.normalize_image(processed)
        standardized = self.preprocessor.standardize_image(normalized)
        
        # Convert to tensor
        tensor = torch.from_numpy(standardized.transpose(2, 0, 1)).float()
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor, image_rgb
    
    def predict(
        self,
        image_path: str,
        return_heatmap: bool = False
    ) -> Dict:
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to image
            return_heatmap: Whether to generate Grad-CAM heatmap
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        tensor, original_image = self.preprocess_image(image_path)
        
        # Run inference
        if self.use_onnx:
            probabilities = self._predict_onnx(tensor)
        else:
            probabilities = self._predict_pytorch(tensor)
        
        # Get prediction
        predicted_class = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        
        # Generate heatmap if requested
        heatmap = None
        overlay = None
        if return_heatmap and not self.use_onnx:
            tensor_cuda = tensor.to(self.device)
            heatmap, overlay = self.gradcam.explain(
                tensor_cuda,
                original_image,
                predicted_class
            )['heatmap'], self.gradcam.explain(
                tensor_cuda,
                original_image,
                predicted_class
            )['overlay']
        
        # Determine referral status
        referral_needed = predicted_class >= 3
        
        result = {
            'predicted_class': predicted_class,
            'label': self.DR_LABELS[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities.tolist(),
            'referral_needed': referral_needed,
            'image_path': image_path
        }
        
        if heatmap is not None:
            result['heatmap'] = heatmap
            result['overlay'] = overlay
        
        return result
    
    def _predict_pytorch(self, tensor: torch.Tensor) -> np.ndarray:
        """Run inference using PyTorch model."""
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
        
        return probabilities.cpu().numpy()[0]
    
    def _predict_onnx(self, tensor: torch.Tensor) -> np.ndarray:
        """Run inference using ONNX model."""
        # Convert to numpy
        input_data = tensor.numpy()
        
        # Get input name
        input_name = self.onnx_session.get_inputs()[0].name
        
        # Run inference
        outputs = self.onnx_session.run(None, {input_name: input_data})
        
        # Apply softmax
        logits = outputs[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        return probabilities[0]
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Make predictions on multiple images.
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append({'error': str(e), 'image_path': path})
        
        return results
    
    @staticmethod
    def save_heatmap(heatmap: np.ndarray, output_path: str):
        """
        Save heatmap as image.
        
        Args:
            heatmap: Heatmap array (H, W) in [0, 1]
            output_path: Output file path
        """
        # Convert to uint8
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Save
        cv2.imwrite(output_path, heatmap_colored)
        print(f"Heatmap saved to {output_path}")
    
    @staticmethod
    def save_overlay(overlay: np.ndarray, output_path: str):
        """
        Save overlay image.
        
        Args:
            overlay: Overlay array (H, W, 3) in [0, 1]
            output_path: Output file path
        """
        # Convert to uint8
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        overlay_rgb = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)
        
        # Save
        cv2.imwrite(output_path, overlay_rgb)
        print(f"Overlay saved to {output_path}")
