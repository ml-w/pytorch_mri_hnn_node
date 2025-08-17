# MRI Node Segmentation Validator

A PyTorch-based framework for training and validating MRI lymph node classification models, with support for torchio data loading and Hugging Face trainer integration.

## Features
- **Modern Data Pipeline**: torchio-based data loading with NIfTI file support and CSV-based label management
- **Hugging Face Integration**: Training with Hugging Face Trainer for robust model training
- **Flexible Architecture**: Support for various 3D CNN architectures with attention mechanisms
- **Comprehensive Evaluation**: Detailed metrics, visualization, and reporting tools
- **Easy Inference**: Batch and single-image inference with multiple output formats
- **Configuration-Driven**: YAML-based configuration for reproducible experiments
- **Production Ready**: Docker support and comprehensive testing

## New Project Structure
```
mri_node_validator/
├── configs/                 # Configuration files
│   ├── training_config.yaml
│   ├── inference_config.yaml
│   └── example_labels.csv
├── data/                    # Data loading and preprocessing
│   ├── dataset.py           # MRI dataset with torchio integration
│   ├── transforms.py        # Data preprocessing and augmentation
│   └── loader.py            # Legacy loader (backward compatibility)
├── models/                  # Model architectures
│   └── detector.py          # PyTorch Lightning model
├── scripts/                 # Training and inference scripts
│   ├── train.py             # Training script with Hugging Face Trainer
│   └── infer.py             # Inference script
├── evaluation/              # Evaluation and metrics
│   ├── evaluator.py         # Model evaluation utilities
│   ├── metrics.py           # Classification metrics
│   └── visualization.py     # Legacy visualization
├── visualization/           # Visualization tools
│   └── plotter.py           # Plotting and visualization utilities
├── tests/                   # Unit tests
├── Dockerfile               # Container configuration
├── requirements.txt         # Python dependencies
└── validator.py             # Legacy validator (backward compatibility)
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU support)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Development Installation
```bash
git clone <repository-url>
cd pytorch_mri_hnn_node
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

Create a CSV file with your labels:
```csv
file_key,label
patient_001_t1,0
patient_002_t1,1
patient_003_t1,0
```

Organize your NIfTI files:
```
data/
├── patient_001_t1.nii.gz
├── patient_002_t1.nii.gz
└── patient_003_t1.nii.gz
```

### 2. Training

```bash
python mri_node_validator/scripts/train.py \
    --config mri_node_validator/configs/training_config.yaml \
    --data_dir /path/to/nifti/files \
    --csv_path /path/to/labels.csv \
    --output_dir ./results \
    --do_train \
    --do_eval
```

### 3. Inference

```bash
python mri_node_validator/scripts/infer.py \
    --model_path ./results/pytorch_model.bin \
    --config mri_node_validator/configs/inference_config.yaml \
    --csv_path /path/to/test_files.csv \
    --data_dir /path/to/nifti/files \
    --output_path predictions.json
```

## Configuration

### Training Configuration
Edit `configs/training_config.yaml` to customize:
- Model architecture and hyperparameters
- Data preprocessing and augmentation
- Training parameters (epochs, learning rate, etc.)
- Evaluation and checkpointing settings

### Inference Configuration
Edit `configs/inference_config.yaml` to customize:
- Model loading settings
- Preprocessing parameters (must match training)
- Output format and content
- Batch processing settings

## Data Format

### CSV Label File
The CSV file should contain at least two columns:
- `file_key`: Key extracted from NIfTI filename (configurable pattern)
- `label`: Integer class label (0, 1, 2, ...)

### NIfTI Files
- Standard NIfTI format (`.nii` or `.nii.gz`)
- 3D volumes supported
- Filename should match the pattern defined in config (default: extracts basename without extension)

## Advanced Usage

### Custom Model Architecture
Modify `models/detector.py` to implement custom architectures:

```python
class CustomDetector(NodeDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom architecture implementation
```

### Custom Data Transforms
Add custom transforms in `data/transforms.py`:

```python
def custom_transform():
    return tio.Lambda(lambda x: your_custom_function(x))
```

### Programmatic Usage

```python
from mri_node_validator.data import MRIDataModule
from mri_node_validator.scripts.train import MRITrainer

# Setup data
data_module = MRIDataModule(
    data_dir="path/to/data",
    csv_path="path/to/labels.csv"
)

# Train model
trainer = MRITrainer(model=model, args=training_args)
trainer.train()
```

## Key Components

### Data Loading Pipeline
The new data loading pipeline uses torchio for robust medical image processing:

- **MRIDataset**: Loads NIfTI files and matches them with CSV labels using configurable key patterns
- **MRIDataModule**: Handles train/val/test splits and data loading with transforms
- **Transforms**: Comprehensive preprocessing including resampling, normalization, and augmentation

### Training Script
The training script (`scripts/train.py`) integrates with Hugging Face Trainer:

- Supports arbitrary model architectures
- Configuration-driven training parameters
- Automatic checkpointing and evaluation
- Comprehensive metrics logging

### Inference Script
The inference script (`scripts/infer.py`) provides flexible inference options:

- Single image or batch processing
- CSV-based file specification
- Multiple output formats (JSON, CSV)
- Configurable preprocessing pipeline

## Legacy Support

The original validator interface is still available for backward compatibility:

```python
from mri_node_validator import Validator

validator = Validator()
results = validator.validate(volume_path, segmentation_path)
```

## Docker Support

```bash
# Build image
docker build -t mri-validator .

# Run training
docker run --gpus all -v /path/to/data:/data mri-validator \
    python scripts/train.py --data_dir /data --csv_path /data/labels.csv
```

## Examples

### Training Example
```bash
# Train a model with custom configuration
python mri_node_validator/scripts/train.py \
    --config configs/training_config.yaml \
    --data_dir /data/mri_scans \
    --csv_path /data/labels.csv \
    --output_dir ./experiments/run_001 \
    --do_train \
    --do_eval \
    --overwrite_output_dir
```

### Inference Examples
```bash
# Inference on single image
python mri_node_validator/scripts/infer.py \
    --model_path ./experiments/run_001/pytorch_model.bin \
    --image_path /data/test_image.nii.gz \
    --output_path single_prediction.json

# Batch inference from directory
python mri_node_validator/scripts/infer.py \
    --model_path ./experiments/run_001/pytorch_model.bin \
    --image_dir /data/test_images \
    --output_path batch_predictions.csv \
    --output_format csv

# CSV-based inference
python mri_node_validator/scripts/infer.py \
    --model_path ./experiments/run_001/pytorch_model.bin \
    --csv_path /data/test_files.csv \
    --data_dir /data/test_images \
    --output_path csv_predictions.json
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `pytest`
5. Submit a pull request

## License

[Add your license information here]