import torch
import torchio as tio
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Dict

class DataLoader:
    """Handles loading and preprocessing of MRI data for classification."""
    
    def __init__(self,
                 normalize: bool = True,
                 resample_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 augmentation: bool = False):
        """
        Args:
            normalize: Whether to normalize image intensities
            resample_spacing: Target voxel spacing in mm (x,y,z)
            augmentation: Whether to apply data augmentation
        """
        self.normalize = normalize
        self.resample_spacing = resample_spacing
        self.augmentation = augmentation
        
        # Define preprocessing transforms
        self.preprocess = tio.Compose([
            tio.Resample(self.resample_spacing),
            tio.CropOrPad((128, 128, 128)) if not augmentation else tio.Compose([])
        ])
        
        # Define augmentation transforms
        self.augment = tio.Compose([
            tio.RandomAffine(),
            tio.RandomNoise(),
            tio.RandomBlur()
        ]) if augmentation else None
        
    def load_and_preprocess(self,
                          image_path: Union[str, Path],
                          label_path: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess image and classification label.
        
        Args:
            image_path: Path to MRI volume (NIfTI)
            label_path: Path to label file (JSON/CSV/TXT) containing binary label (0/1)
            
        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        # Load image
        image = tio.ScalarImage(image_path)
        
        # Load label
        label = self._load_label(label_path)
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        # Apply preprocessing
        subject = tio.Subject(image=image)
        subject = self.preprocess(subject)
        
        # Apply augmentation if enabled
        if self.augmentation:
            subject = self.augment(subject)
            
        # Convert to tensors and normalize
        image_tensor = subject.image.data
        if self.normalize:
            image_tensor = (image_tensor - image_tensor.mean()) / image_tensor.std()
            
        return image_tensor, label_tensor
    
    def _load_label(self, label_path: Union[str, Path]) -> int:
        """Load binary classification label from file.
        
        Args:
            label_path: Path to label file (JSON/CSV/TXT)
            
        Returns:
            Binary label (0 or 1)
        """
        if str(label_path).endswith('.json'):
            with open(label_path) as f:
                data = json.load(f)
                return int(data['label'])
        elif str(label_path).endswith('.csv'):
            with open(label_path) as f:
                return int(f.read().strip())
        elif str(label_path).endswith('.txt'):
            with open(label_path) as f:
                return int(f.read().strip())
        else:
            raise ValueError(f"Unsupported label file format: {label_path}")