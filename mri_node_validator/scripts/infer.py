#!/usr/bin/env python3
"""
Inference script for trained MRI classification models.
"""

import os
import sys
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import MRIDataset
from data.transforms import get_inference_transforms
from models.detector import NodeDetector
import torchio as tio


class MRIInferencer:
    """Inference class for MRI classification models."""
    
    def __init__(
        self,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the inferencer.
        
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration dictionary
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.model_path = model_path
        self.config = config or {}
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Setup transforms
        self.transforms = self._setup_transforms()
        
    def _load_model(self) -> NodeDetector:
        """Load the trained model."""
        print(f"Loading model from {self.model_path}")
        
        if self.model_path.endswith('.ckpt'):
            # PyTorch Lightning checkpoint
            model = NodeDetector.load_from_checkpoint(self.model_path)
        else:
            # Regular PyTorch checkpoint
            model = NodeDetector(**self.config.get('model', {}))
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        
        model.to(self.device)
        return model
    
    def _setup_transforms(self) -> tio.Compose:
        """Setup preprocessing transforms for inference."""
        preprocessing_config = self.config.get('preprocessing', {})
        
        target_spacing = preprocessing_config.get('target_spacing', (1.0, 1.0, 1.0))
        target_shape = preprocessing_config.get('target_shape', (128, 128, 128))
        
        return get_inference_transforms(
            target_spacing=target_spacing,
            target_shape=target_shape
        )
    
    def predict_single(
        self,
        image_path: Union[str, Path],
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Predict on a single image.
        
        Args:
            image_path: Path to NIfTI image file
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary containing prediction results
        """
        # Load and preprocess image
        image = tio.ScalarImage(image_path)
        subject = tio.Subject({'image': image})
        
        if self.transforms:
            subject = self.transforms(subject)
        
        # Prepare input tensor
        image_tensor = subject['image'].data.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor, image_tensor)  # Using same tensor for both inputs
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Get probabilities and predictions
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        result = {
            'file_path': str(image_path),
            'predicted_class': predicted_class.cpu().item(),
            'confidence': probabilities.max().cpu().item()
        }
        
        if return_probabilities:
            result['probabilities'] = probabilities.cpu().numpy().tolist()[0]
        
        return result
    
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 8,
        return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict on a batch of images.
        
        Args:
            image_paths: List of paths to NIfTI image files
            batch_size: Batch size for inference
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            # Load and preprocess batch
            for image_path in batch_paths:
                image = tio.ScalarImage(image_path)
                subject = tio.Subject({'image': image})
                
                if self.transforms:
                    subject = self.transforms(subject)
                
                batch_tensors.append(subject['image'].data)
            
            # Stack tensors
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(batch_tensor, batch_tensor)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Get probabilities and predictions
                probabilities = torch.softmax(logits, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
            
            # Process results
            for j, image_path in enumerate(batch_paths):
                result = {
                    'file_path': str(image_path),
                    'predicted_class': predicted_classes[j].cpu().item(),
                    'confidence': probabilities[j].max().cpu().item()
                }
                
                if return_probabilities:
                    result['probabilities'] = probabilities[j].cpu().numpy().tolist()
                
                results.append(result)
        
        return results
    
    def predict_from_csv(
        self,
        data_dir: Union[str, Path],
        csv_path: Union[str, Path],
        key_pattern: str = r"(.+)\.nii(?:\.gz)?$",
        file_key_column: str = "file_key",
        batch_size: int = 8,
        return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict on images specified in a CSV file.
        
        Args:
            data_dir: Directory containing NIfTI files
            csv_path: Path to CSV file with file keys
            key_pattern: Regex pattern to match filenames
            file_key_column: Column name for file keys in CSV
            batch_size: Batch size for inference
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        # Create dataset without labels (for inference)
        dataset = MRIDataset(
            data_dir=data_dir,
            csv_path=csv_path,
            key_pattern=key_pattern,
            transforms=self.transforms,
            file_key_column=file_key_column,
            label_column=file_key_column  # Dummy column, won't be used
        )
        
        # Get all file paths
        image_paths = dataset.file_paths
        
        return self.predict_batch(
            image_paths=image_paths,
            batch_size=batch_size,
            return_probabilities=return_probabilities
        )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_results(
    results: List[Dict[str, Any]],
    output_path: str,
    format: str = 'json'
):
    """
    Save prediction results to file.
    
    Args:
        results: List of prediction results
        output_path: Output file path
        format: Output format ('json', 'csv')
    """
    if format.lower() == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif format.lower() == 'csv':
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {format}")
    
    print(f"Results saved to {output_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference with trained MRI classifier")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_path", type=str, help="Path to single NIfTI image")
    input_group.add_argument("--image_dir", type=str, help="Directory containing NIfTI images")
    input_group.add_argument("--csv_path", type=str, help="CSV file with image paths/keys")
    
    # Additional arguments
    parser.add_argument("--data_dir", type=str, help="Data directory (required when using --csv_path)")
    parser.add_argument("--output_path", type=str, default="predictions.json", help="Output file path")
    parser.add_argument("--output_format", type=str, default="json", choices=["json", "csv"], help="Output format")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--device", type=str, help="Device to use (cpu, cuda)")
    parser.add_argument("--key_pattern", type=str, default=r"(.+)\.nii(?:\.gz)?$", help="Regex pattern for filename keys")
    parser.add_argument("--file_key_column", type=str, default="file_key", help="Column name for file keys in CSV")
    parser.add_argument("--no_probabilities", action="store_true", help="Don't return class probabilities")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Initialize inferencer
    inferencer = MRIInferencer(
        model_path=args.model_path,
        config=config,
        device=args.device
    )
    
    # Run inference based on input type
    if args.image_path:
        # Single image inference
        print(f"Running inference on single image: {args.image_path}")
        result = inferencer.predict_single(
            image_path=args.image_path,
            return_probabilities=not args.no_probabilities
        )
        results = [result]
        
    elif args.image_dir:
        # Directory inference
        print(f"Running inference on directory: {args.image_dir}")
        image_dir = Path(args.image_dir)
        image_paths = []
        
        for ext in ['.nii', '.nii.gz']:
            image_paths.extend(image_dir.rglob(f'*{ext}'))
        
        if not image_paths:
            print("No NIfTI files found in directory")
            return
        
        print(f"Found {len(image_paths)} images")
        results = inferencer.predict_batch(
            image_paths=image_paths,
            batch_size=args.batch_size,
            return_probabilities=not args.no_probabilities
        )
        
    elif args.csv_path:
        # CSV-based inference
        if not args.data_dir:
            parser.error("--data_dir is required when using --csv_path")
        
        print(f"Running inference from CSV: {args.csv_path}")
        results = inferencer.predict_from_csv(
            data_dir=args.data_dir,
            csv_path=args.csv_path,
            key_pattern=args.key_pattern,
            file_key_column=args.file_key_column,
            batch_size=args.batch_size,
            return_probabilities=not args.no_probabilities
        )
    
    # Save results
    save_results(results, args.output_path, args.output_format)
    
    # Print summary
    print(f"\nInference completed on {len(results)} images")
    if results:
        class_counts = {}
        for result in results:
            pred_class = result['predicted_class']
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        
        print("Prediction summary:")
        for class_id, count in sorted(class_counts.items()):
            print(f"  Class {class_id}: {count} images")


if __name__ == "__main__":
    main()