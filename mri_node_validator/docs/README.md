# MRI Node Segmentation Validator Documentation

## Overview
The MRI Node Segmentation Validator is a PyTorch-based framework for detecting when lymph node segmentation masks erroneously merge multiple nodes into a single segmentation. The system combines 3D CNNs with transformer components to analyze both local node characteristics and contextual information.

## Key Features
- **Multi-scale Feature Extraction**: Captures both fine-grained node details and broader contextual relationships
- **Boundary Attention**: Focuses on regions between potentially merged nodes using attention mechanisms
- **Flexible Input Support**: Processes both segmentation masks and JSON bounding box annotations
- **Comprehensive Evaluation**: Provides multiple metrics for segmentation quality assessment
- **Interactive Visualization**: Streamlit-based tools for result analysis and debugging

## Installation
```bash
git clone https://github.com/your-repo/mri_node_validator.git
cd mri_node_validator
pip install -r requirements.txt
```

## Usage
### Basic Validation
```python
from mri_node_validator import Validator

validator = Validator(model_path='path/to/model.pt')
results = validator.validate('path/to/volume.nii', 'path/to/segmentation.nii')
```

### Visualization
```bash
streamlit run mri_node_validator/evaluation/visualization.py
```

## API Reference
### `Validator` Class
- `validate(volume_path, segmentation_path)`: Runs validation on input volume and segmentation
- `generate_report(results, output_path)`: Creates PDF report with validation metrics

### Data Loading
- Supports NIfTI, DICOM, and JSON bounding box formats via TorchIO

## Evaluation Metrics
- **Merge Detection Accuracy**: Measures correct identification of merged nodes
- **Boundary Precision**: Quantifies boundary segmentation quality
- **Spatial Consistency**: Evaluates geometric relationships between nodes

## Contributing
Pull requests welcome. Please ensure:
1. Tests pass (`pytest tests/`)
2. Code is formatted (`black .`)
3. Type hints are maintained