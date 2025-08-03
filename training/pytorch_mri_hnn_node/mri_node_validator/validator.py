import torch
from typing import Union, Dict, List, Optional
from pathlib import Path
import numpy as np

from .data.loader import DataLoader
from .models.detector import NodeDetector
from .metrics.evaluator import Evaluator
from .visualization.plotter import Plotter

class Validator:
    """Main class for validating MRI lymph node segmentations."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the validator with optional configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.data_loader = DataLoader(**self.config.get("data", {}))
        self.detector = NodeDetector(**self.config.get("model", {}))
        self.evaluator = Evaluator(**self.config.get("metrics", {}))
        self.plotter = Plotter(**self.config.get("visualization", {}))
        
    def _default_config(self) -> Dict:
        """Return default configuration parameters."""
        return {
            "data": {
                "normalize": True,
                "resample_spacing": [1.0, 1.0, 1.0],
                "augmentation": False
            },
            "model": {
                "model_type": "3d_resnet_attention",
                "pretrained": False
            },
            "metrics": {
                "iou_threshold": 0.5,
                "distance_threshold": 5.0
            },
            "visualization": {
                "output_dir": "results",
                "interactive": True
            }
        }
        
    def validate(self, 
                image_path: Union[str, Path], 
                segmentation_path: Union[str, Path]) -> Dict:
        """Validate a segmentation against ground truth.
        
        Args:
            image_path: Path to MRI volume
            segmentation_path: Path to segmentation mask or JSON with bboxes
            
        Returns:
            Dictionary containing validation results and metrics
        """
        # Load and preprocess data
        image, segmentation = self.data_loader.load_and_preprocess(
            image_path, segmentation_path)
            
        # Run detection
        predictions = self.detector.predict(image, segmentation)
        
        # Evaluate results
        metrics = self.evaluator.evaluate(predictions, segmentation)
        
        # Generate visualizations
        self.plotter.generate_plots(image, segmentation, predictions, metrics)
        
        return {
            "predictions": predictions,
            "metrics": metrics,
            "visualizations": self.plotter.output_files
        }