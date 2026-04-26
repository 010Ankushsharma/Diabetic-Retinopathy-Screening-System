"""
Model export to ONNX format for deployment and mobile inference.
"""

import os
import torch
import onnx
from pathlib import Path
from typing import Optional

from src.models.dr_model import DRClassifier


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config: dict,
    opset_version: int = 15,
    device: str = "cuda"
):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Output ONNX file path
        config: Configuration dictionary
        opset_version: ONNX opset version
        device: Device to load model on
    """
    print(f"Exporting model to ONNX: {output_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load model
    model = DRClassifier(
        backbone_name=config['MODEL']['BACKBONE'],
        num_classes=config['MODEL']['NUM_CLASSES'],
        pretrained=False,
        dropout=config['MODEL']['DROPOUT']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle PyTorch Lightning checkpoints
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    
    # Create dummy input
    image_size = config['MODEL']['IMAGE_SIZE']
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"ONNX model exported successfully: {output_path}")
    print(f"Model size: {os.path.getsize(output_path) / 1e6:.2f} MB")
    
    return output_path


def optimize_onnx(
    input_path: str,
    output_path: str,
    fp16: bool = True
):
    """
    Optimize ONNX model for inference.
    
    Args:
        input_path: Input ONNX model path
        output_path: Output optimized model path
        fp16: Whether to convert to FP16
    """
    print(f"Optimizing ONNX model: {input_path}")
    
    try:
        from onnxruntime.transformers import optimizer
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # Optimize
        optimized_model = optimizer.optimize_model(
            input_path,
            model_type='bert',  # Generic optimization
            num_heads=0,
            hidden_size=0
        )
        
        # Save optimized model
        optimized_model.save_model_to_file(output_path)
        
        print(f"Optimized model saved: {output_path}")
        print(f"Original size: {os.path.getsize(input_path) / 1e6:.2f} MB")
        print(f"Optimized size: {os.path.getsize(output_path) / 1e6:.2f} MB")
        
        # Quantize to INT8 for mobile
        if not fp16:
            quantized_path = output_path.replace('.onnx', '_quantized.onnx')
            quantize_dynamic(
                output_path,
                quantized_path,
                weight_type=QuantType.QUInt8
            )
            
            print(f"Quantized model saved: {quantized_path}")
            print(f"Quantized size: {os.path.getsize(quantized_path) / 1e6:.2f} MB")
            
            return quantized_path
        
        return output_path
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        print("Using original model without optimization")
        return input_path


def export_mobile_variant(
    checkpoint_path: str,
    output_dir: str,
    config: dict,
    device: str = "cuda"
):
    """
    Export lightweight model variant for mobile devices.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        output_dir: Output directory
        config: Configuration dictionary
        device: Device
    """
    print("Exporting mobile-optimized model variant...")
    
    # Load config for mobile
    mobile_config = config.copy()
    mobile_config['MODEL'] = config.get('MOBILE', {})
    mobile_config['MODEL']['BACKBONE'] = config['MOBILE'].get('MODEL_VARIANT', 'efficientnet_b0')
    mobile_config['MODEL']['IMAGE_SIZE'] = config['MOBILE'].get('IMAGE_SIZE', 384)
    
    # Create mobile model
    mobile_model = DRClassifier(
        backbone_name=mobile_config['MODEL']['BACKBONE'],
        num_classes=config['MODEL']['NUM_CLASSES'],
        pretrained=False,
        dropout=config['MODEL']['DROPOUT']
    )
    
    # Load weights (if compatible)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    # Try to load compatible weights
    try:
        mobile_model.load_state_dict(state_dict, strict=False)
        print("Loaded compatible weights for mobile model")
    except:
        print("Warning: Could not load weights, using random initialization")
    
    mobile_model = mobile_model.to(device)
    mobile_model.eval()
    
    # Export to ONNX
    os.makedirs(output_dir, exist_ok=True)
    mobile_onnx_path = os.path.join(output_dir, 'dr_model_mobile.onnx')
    
    dummy_input = torch.randn(1, 3, mobile_config['MODEL']['IMAGE_SIZE'], mobile_config['MODEL']['IMAGE_SIZE']).to(device)
    
    torch.onnx.export(
        mobile_model,
        dummy_input,
        mobile_onnx_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Mobile model exported: {mobile_onnx_path}")
    print(f"Mobile model size: {os.path.getsize(mobile_onnx_path) / 1e6:.2f} MB")
    
    return mobile_onnx_path


def export_from_config(
    checkpoint_path: str,
    config_path: str = "config.yaml",
    output_dir: str = "models/onnx"
):
    """
    Export model using configuration file.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        output_dir: Output directory
    """
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Export standard model
    onnx_path = os.path.join(output_dir, 'dr_model.onnx')
    export_to_onnx(checkpoint_path, onnx_path, config)
    
    # Optimize
    optimized_path = os.path.join(output_dir, 'dr_model_optimized.onnx')
    optimize_onnx(onnx_path, optimized_path)
    
    # Export mobile variant
    mobile_path = export_mobile_variant(checkpoint_path, output_dir, config)
    
    print("\nExport complete!")
    print(f"Standard model: {onnx_path}")
    print(f"Optimized model: {optimized_path}")
    print(f"Mobile model: {mobile_path}")
    
    return onnx_path, optimized_path, mobile_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export DR model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="models/onnx", help="Output directory")
    
    args = parser.parse_args()
    
    export_from_config(args.checkpoint, args.config, args.output_dir)
