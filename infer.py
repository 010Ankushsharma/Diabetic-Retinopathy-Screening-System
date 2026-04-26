"""
Example inference script for DR classification.
Usage: python infer.py --image path/to/image.jpg
"""

import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path

from src.inference.inference import DRInference


def main():
    parser = argparse.ArgumentParser(description='DR Inference Script')
    parser.add_argument('--image', type=str, required=True, help='Path to retinal image')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--heatmap', action='store_true', help='Generate Grad-CAM heatmap')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine model path
    model_path = args.model or config.get('MODEL_PATH', 'models/checkpoints/best_model.ckpt')
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first or download pre-trained weights")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from: {model_path}")
    print(f"Processing image: {args.image}")
    
    # Initialize inference engine
    inference = DRInference(
        model_path=model_path,
        config=config,
        use_onnx=False
    )
    
    # Make prediction
    result = inference.predict(args.image, return_heatmap=args.heatmap)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"DR Grade: {result['predicted_class']} - {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Referral Needed: {'YES ⚠️' if result['referral_needed'] else 'No'}")
    
    print("\nClass Probabilities:")
    for i, prob in enumerate(result['probabilities']):
        bar = "█" * int(prob * 30)
        print(f"  Class {i} ({inference.DR_LABELS[i]:20s}): {prob:.4f} {bar}")
    
    # Save heatmap if requested
    if args.heatmap and 'overlay' in result:
        overlay_path = output_dir / "heatmap_overlay.png"
        heatmap_path = output_dir / "heatmap.png"
        
        inference.save_overlay(result['overlay'], str(overlay_path))
        inference.save_heatmap(result['heatmap'], str(heatmap_path))
        
        print(f"\nHeatmap saved to: {overlay_path}")
        print(f"Heatmap only: {heatmap_path}")
    
    # Save original image with prediction info
    image = cv2.imread(args.image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Add text overlay
    text = f"{result['label']} ({result['confidence']:.2%})"
    cv2.putText(
        image,
        text,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if result['predicted_class'] < 3 else (0, 0, 255),
        2
    )
    
    output_image_path = output_dir / "prediction_result.png"
    cv2.imwrite(str(output_image_path), image)
    print(f"Result image saved to: {output_image_path}")
    
    print("\n" + "="*60)
    
    # Recommendation
    if result['referral_needed']:
        print("⚠️  RECOMMENDATION: URGENT referral to ophthalmologist required")
    else:
        print("✓ RECOMMENDATION: Continue regular screening schedule")
    
    print("="*60)


if __name__ == "__main__":
    main()
