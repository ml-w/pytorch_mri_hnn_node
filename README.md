# PyTorch MRI HNN Node Classification

A comprehensive PyTorch-based framework for training and validating MRI lymph node classification models with modern data loading pipelines and Hugging Face integration.

## Overview

This project provides a complete solution for MRI-based lymph node classification, featuring:

- **Modern Data Pipeline**: torchio-based data loading with NIfTI file support
- **CSV-Based Label Management**: Flexible label assignment using file key matching
- **Hugging Face Integration**: Robust training with Hugging Face Trainer
- **Comprehensive Evaluation**: Detailed metrics and visualization tools
- **Production-Ready Inference**: Batch and single-image inference capabilities

## Project Structure

```
pytorch_mri_hnn_node/
├── mri_node_validator/          # Main package
│   ├── configs/                 # Configuration files
│   ├── data/                    # Data loading and preprocessing
│   ├── models/                  # Model architectures
│   ├── scripts/                 # Training and inference scripts
│   ├── evaluation/              # Evaluation and metrics
│   ├── visualization/           # Visualization tools
│   └── tests/                   # Unit tests
└── README.md                    # This file
```

## Quick Start

### Installation

```bash
git clone <repository-url>
cd pytorch_mri_hnn_node
pip install -r mri_node_validator/requirements.txt
```

### Data Preparation

1. **Organize your NIfTI files**:
```
data/
├── patient_001_t1.nii.gz
├── patient_002_t1.nii.gz
└── patient_003_t1.nii.gz
```

2. **Create a CSV file with labels**:
```csv
file_key,label
patient_001_t1,0
patient_002_t1,1
patient_003_t1,0
```

### Training

```bash
python mri_node_validator/scripts/train.py \
    --config mri_node_validator/configs/training_config.yaml \
    --data_dir /path/to/nifti/files \
    --csv_path /path/to/labels.csv \
    --output_dir ./results \
    --do_train \
    --do_eval
```

### Inference

```bash
python mri_node_validator/scripts/infer.py \
    --model_path ./results/pytorch_model.bin \
    --config mri_node_validator/configs/inference_config.yaml \
    --csv_path /path/to/test_files.csv \
    --data_dir /path/to/nifti/files \
    --output_path predictions.json
```

## Key Features

### 🔄 Modern Data Loading Pipeline

- **torchio Integration**: Robust medical image processing with torchio
- **CSV-Based Labels**: Flexible label management using file key matching
- **Configurable Preprocessing**: Resampling, normalization, and augmentation
- **Automatic Data Splits**: Train/validation/test splitting with reproducible seeds

### 🤗 Hugging Face Integration

- **Trainer API**: Leverage Hugging Face's robust training infrastructure
- **Configuration-Driven**: YAML-based configuration for all parameters
- **Automatic Checkpointing**: Model saving and resuming capabilities
- **Comprehensive Logging**: TensorBoard and other logging integrations

### 📊 Comprehensive Evaluation

- **Multiple Metrics**: Accuracy, precision, recall, F1-score, AUC
- **Visualization Tools**: Confusion matrices, ROC curves, sample predictions
- **Interactive Dashboards**: Plotly-based interactive visualizations
- **Detailed Reports**: Automated report generation

### 🚀 Production-Ready Inference

- **Flexible Input**: Single images, directories, or CSV-specified files
- **Batch Processing**: Efficient batch inference with configurable batch sizes
- **Multiple Formats**: JSON and CSV output formats
- **Memory Optimization**: Automatic mixed precision and memory management

## Data Format Specification

### CSV Label File Format

The CSV file must contain at least these columns:
- `file_key`: Identifier extracted from NIfTI filename
- `label`: Integer class label (0, 1, 2, ...)

Example:
```csv
file_key,label,additional_info
patient_001_t1,0,normal
patient_002_t1,1,merged_nodes
patient_003_t1,0,normal
```

### File Key Extraction

By default, the system extracts the basename without extension:
- `patient_001_t1.nii.gz` → `patient_001_t1`
- `scan_123.nii` → `scan_123`

This pattern is configurable via the `key_pattern` parameter in the configuration files.

## Configuration

### Training Configuration (`configs/training_config.yaml`)

Key sections:
- **Model**: Architecture type, learning rate, number of classes
- **Data**: Batch size, splits, preprocessing parameters
- **Training**: Epochs, optimization, scheduling, early stopping
- **Preprocessing**: Spacing, shape, augmentation settings

### Inference Configuration (`configs/inference_config.yaml`)

Key sections:
- **Model**: Model loading parameters
- **Preprocessing**: Must match training preprocessing
- **Inference**: Batch size, device, output settings
- **Output**: Format, content, visualization options

## Advanced Usage

### Custom Model Architectures

Extend the base model class:

```python
from mri_node_validator.models.detector import NodeDetector

class CustomDetector(NodeDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Implement custom architecture
```

### Custom Data Transforms

Add transforms to the preprocessing pipeline:

```python
from mri_node_validator.data.transforms import CustomTransforms

# Add custom preprocessing
custom_transform = CustomTransforms.intensity_clipping(1.0, 99.0)
```

### Programmatic Training

```python
from mri_node_validator.data import MRIDataModule
from mri_node_validator.scripts.train import MRITrainer

# Setup data module
data_module = MRIDataModule(
    data_dir="path/to/data",
    csv_path="path/to/labels.csv",
    batch_size=8
)

# Setup and run training
trainer = MRITrainer(model=model, args=training_args)
trainer.train()
```

## Examples

See the [`mri_node_validator/README.md`](mri_node_validator/README.md) for detailed examples and usage instructions.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchio 0.18.85+
- transformers 4.20.0+
- See [`mri_node_validator/requirements.txt`](mri_node_validator/requirements.txt) for complete list

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mri_node_validator,
  title={PyTorch MRI HNN Node Classification},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}