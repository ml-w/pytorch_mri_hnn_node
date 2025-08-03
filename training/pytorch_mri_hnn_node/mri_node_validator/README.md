# MRI Node Segmentation Validator

A PyTorch-based framework for validating MRI lymph node segmentations, focusing on detecting cases where multiple nodes are incorrectly merged in a single segmentation mask.

## Features
- 3D MRI volume and segmentation mask processing
- Detection of merged node segmentation errors
- Support for both segmentation masks and JSON bounding box inputs
- Deep learning architecture optimized for classification
- Multi-scale feature extraction
- Attention mechanisms for boundary regions
- Comprehensive evaluation metrics
- Docker support
- Visualization tools

## Project Structure
```
mri_node_validator/
├── configs/         # Configuration files
├── data/            # Data loading and preprocessing
├── models/          # Model architectures
├── metrics/         # Evaluation metrics
├── visualization/   # Visualization tools  
├── utils/           # Utility functions
├── tests/           # Unit tests
├── Dockerfile       # Container configuration
└── requirements.txt # Python dependencies
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from mri_node_validator import Validator

validator = Validator()
results = validator.validate(volume_path, segmentation_path)