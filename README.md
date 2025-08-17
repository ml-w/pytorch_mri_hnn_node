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

## Hugging Face Integration Tutorial

This tutorial provides a comprehensive guide for setting up training and inference using custom models pulled from Hugging Face. The framework provides seamless integration with Hugging Face's ecosystem, allowing you to leverage pre-trained models and share your own models with the community.

### Overview of Hugging Face Integration

The project includes a Hugging Face-compatible wrapper (`HFNodeDetector`) that:
- Wraps the custom `NodeDetector` model for compatibility with Hugging Face Trainer
- Supports saving and loading models in Hugging Face format
- Enables easy sharing of models via Hugging Face Hub
- Allows fine-tuning of pre-trained models from Hugging Face Hub

### Prerequisites

Before starting, ensure you have:
- A Hugging Face account (free at [huggingface.co](https://huggingface.co))
- The `huggingface-hub` package installed (included in requirements.txt)
- Git LFS installed for large model files

```bash
# Install Git LFS if not already installed
git lfs install

# Login to Hugging Face (optional, for private models)
huggingface-cli login
```

### Part 1: Training with Custom Models

#### Option A: Training a New Model from Scratch

1. **Prepare your data** as described in the Quick Start section

2. **Configure training settings** in [`configs/training_config.yaml`](mri_node_validator/configs/training_config.yaml):
```yaml
model:
  model_type: "3d_resnet_attention"  # Your custom architecture
  num_classes: 2                     # Number of output classes
  learning_rate: 1e-3

training:
  num_train_epochs: 50
  per_device_train_batch_size: 8
  evaluation_strategy: "epoch"
  save_strategy: "epoch"
  load_best_model_at_end: true
  metric_for_best_model: "eval_f1"
```

3. **Run training**:
```bash
python mri_node_validator/scripts/train.py \
    --config mri_node_validator/configs/training_config.yaml \
    --data_dir /path/to/nifti/files \
    --csv_path /path/to/labels.csv \
    --output_dir ./results \
    --do_train \
    --do_eval
```

#### Option B: Fine-tuning a Pre-trained Model from Hugging Face Hub

1. **Find a suitable model** on Hugging Face Hub or use your own previously trained model

2. **Configure training** to use the pre-trained model:
```bash
python mri_node_validator/scripts/train.py \
    --config mri_node_validator/configs/training_config.yaml \
    --data_dir /path/to/nifti/files \
    --csv_path /path/to/labels.csv \
    --output_dir ./results \
    --model_name_or_path username/model-name \
    --do_train \
    --do_eval
```

3. **Adjust configuration** for fine-tuning:
```yaml
# In training_config.yaml
model:
  model_type: "3d_resnet_attention"  # Must match the pre-trained model architecture
  num_classes: 2                     # Can be different from pre-trained model
  learning_rate: 1e-4                # Lower learning rate for fine-tuning

training:
  num_train_epochs: 20               # Fewer epochs for fine-tuning
  per_device_train_batch_size: 4     # Adjust based on GPU memory
```

### Part 2: Saving and Sharing Models

#### Saving Models in Hugging Face Format

After training, your model is automatically saved in Hugging Face format in the output directory:
```
results/
├── config.json              # Model configuration
├── model.safetensors        # Model weights (newer format)
├── pytorch_model.bin        # Model weights (legacy format)
├── training_args.bin        # Training arguments
└── trainer_state.json       # Trainer state
```

#### Uploading Models to Hugging Face Hub

1. **Initialize a Git repository** for your model:
```bash
cd ./results
git init
git lfs track "*.safetensors"
git lfs track "*.bin"
git add .gitattributes
git commit -m "Initialize model repository"
```

2. **Create a new model repository** on Hugging Face Hub:
```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Create repository (replace username/model-name with your details)
huggingface-cli repo create username/model-name --yes
```

3. **Upload your model**:
```bash
# Add remote
git remote add origin https://huggingface.co/username/model-name

# Add and commit files
git add .
git commit -m "Add trained MRI classification model"

# Push to Hub
git push -u origin main
```

Alternatively, use the Python API:
```python
from huggingface_hub import HfApi, HfFolder

# Initialize API
api = HfApi()

# Upload folder
api.upload_folder(
    folder_path="./results",
    repo_id="username/model-name",
    repo_type="model"
)
```

### Part 3: Inference with Hugging Face Models

#### Option A: Using Models from Hugging Face Hub

1. **Download a model** from Hugging Face Hub:
```bash
# Using git clone (recommended for large models)
git clone https://huggingface.co/username/model-name
cd model-name

# Or using Python API
from huggingface_hub import snapshot_download
snapshot_download(repo_id="username/model-name")
```

2. **Run inference** with the downloaded model:
```bash
python mri_node_validator/scripts/infer.py \
    --model_path ./model-name \
    --config mri_node_validator/configs/inference_config.yaml \
    --image_path /path/to/test_image.nii.gz \
    --output_path predictions.json
```

#### Option B: Direct Inference from Hub (without downloading)

You can also run inference directly by specifying the Hub model ID:

```bash
python mri_node_validator/scripts/infer.py \
    --model_path username/model-name \
    --config mri_node_validator/configs/inference_config.yaml \
    --image_path /path/to/test_image.nii.gz \
    --output_path predictions.json
```

#### Batch Inference

For processing multiple images:

```bash
# Process all images in a directory
python mri_node_validator/scripts/infer.py \
    --model_path username/model-name \
    --config mri_node_validator/configs/inference_config.yaml \
    --image_dir /path/to/test_images \
    --output_path batch_predictions.json \
    --batch_size 16

# Process images specified in CSV
python mri_node_validator/scripts/infer.py \
    --model_path username/model-name \
    --config mri_node_validator/configs/inference_config.yaml \
    --csv_path /path/to/test_files.csv \
    --data_dir /path/to/nifti/files \
    --output_path csv_predictions.json
```

### Part 4: Advanced Usage

#### Custom Model Architectures

To use a custom model architecture with Hugging Face:

1. **Extend the HFNodeDetector class**:
```python
from mri_node_validator.models.hf_detector import HFNodeDetector, MRIConfig

class CustomHFNodeDetector(HFNodeDetector):
    def __init__(self, config: MRIConfig):
        super().__init__(config)
        # Add your custom architecture here
        self.custom_layer = nn.Linear(512, config.num_labels)
    
    def forward(self, image=None, segmentation=None, labels=None, **kwargs):
        # Implement custom forward pass
        outputs = super().forward(image, segmentation, labels, **kwargs)
        # Add custom processing
        return outputs
```

2. **Register your custom model**:
```python
from transformers import AutoModel

AutoModel.register(CustomHFNodeDetector, "custom-mri-detector")
```

#### Model Cards and Documentation

Create comprehensive model cards for your shared models:

```markdown
---
language: en
license: mit
tags:
- medical-imaging
- mri
- classification
- pytorch
---

# MRI Node Classification Model

## Model Description
This model is trained for classifying lymph nodes in MRI scans...

## Intended Uses & Limitations
- **Intended Use**: Classification of lymph nodes in medical MRI scans
- **Limitations**: Trained on specific scanner types and protocols...

## Training Data
- Dataset: [Describe your dataset]
- Preprocessing: [Describe preprocessing steps]
- Augmentation: [List augmentations used]

## Evaluation Results
- Accuracy: 0.95
- F1-Score: 0.94
- AUC: 0.97

## How to Use
```python
from mri_node_validator.models.hf_detector import HFNodeDetector

model = HFNodeDetector.from_pretrained("username/model-name")
# Use model for inference
```
```

#### Version Control and Model Management

1. **Tag your model versions**:
```bash
git tag v1.0.0
git push origin v1.0.0
```

2. **Use model branches** for different experiments:
```bash
git checkout -b experiment/new-architecture
# Make changes and commit
git push origin experiment/new-architecture
```

3. **Model versioning in code**:
```python
# Load specific version
model = HFNodeDetector.from_pretrained("username/model-name", revision="v1.0.0")

# Load from branch
model = HFNodeDetector.from_pretrained("username/model-name", revision="experiment/new-architecture")
```

### Part 5: Troubleshooting

#### Common Issues

1. **Model Loading Errors**:
```bash
# Ensure you have the latest transformers version
pip install --upgrade transformers

# Check model architecture compatibility
# The model_type in config must match the pre-trained model
```

2. **Memory Issues**:
```yaml
# In inference_config.yaml, reduce batch size
inference:
  batch_size: 4
  use_amp: true  # Enable mixed precision
```

3. **Hugging Face Hub Authentication**:
```bash
# Check login status
huggingface-cli whoami

# Re-login if needed
huggingface-cli logout
huggingface-cli login
```

#### Performance Optimization

1. **Enable mixed precision** for faster inference:
```yaml
# In inference_config.yaml
inference:
  use_amp: true
  device: "cuda"
```

2. **Optimize data loading**:
```yaml
# In inference_config.yaml
data:
  num_workers: 4
  pin_memory: true
```

3. **Use torch.compile** for PyTorch 2.0+:
```yaml
# In inference_config.yaml
performance:
  torch_compile: true
```

### Best Practices

1. **Always test locally** before uploading to Hub
2. **Include comprehensive model cards** with usage examples
3. **Use semantic versioning** for your model releases
4. **Provide example scripts** for easy reproduction
5. **Document preprocessing requirements** clearly
6. **Include sample predictions** for verification
7. **Test inference on multiple devices** (CPU, GPU)

### Community and Support

- **Hugging Face Community**: [Join the discussion](https://discuss.huggingface.co/)
- **Model Issues**: Report issues on the model repository
- **Framework Issues**: Report issues on this project's repository
- **Documentation**: Check the [Hugging Face documentation](https://huggingface.co/docs/transformers/)

This tutorial provides a complete workflow for leveraging Hugging Face integration with your MRI classification models. For additional examples and advanced use cases, refer to the Hugging Face documentation and the project's test files.

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
