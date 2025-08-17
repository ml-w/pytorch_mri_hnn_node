# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PyTorch-based MRI lymph node classification framework with Hugging Face integration. The codebase focuses on detecting merged lymph nodes from 3D MRI volumes using CSV-based label management and modern data loading pipelines.

## Common Commands

### Installation and Setup
```bash
pip install -r mri_node_validator/requirements.txt
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

### Testing
```bash
# Run all tests from the repository root
cd /Users/lwong/Source/pytorch_mri_hnn_node
python -m pytest mri_node_validator/tests/

# Run specific test file
python -m pytest mri_node_validator/tests/test_detector.py

# Run with verbose output
python -m pytest mri_node_validator/tests/ -v
```

### Code Quality (from requirements.txt)
```bash
# Format code
black mri_node_validator/

# Type checking
mypy mri_node_validator/

# Linting
pylint mri_node_validator/
```

### Example Scripts
```bash
# Run example scripts to learn the framework
cd examples/

# 1. Minimal usage example
python 01_minimal_usage.py

# 2. Basic training pipeline
python 02_training_basic.py

# 3. Detailed inference with evaluation
python 03_inference_detailed.py

# 4. Advanced training with custom augmentation
python 04_custom_augmentation.py

# 5. Hugging Face Hub integration
python 05_huggingface_hub_integration.py
```

## Architecture Overview

### Core Components

- **Data Pipeline**: `mri_node_validator/data/` - torchio-based NIfTI loading with CSV label management
- **Models**: `mri_node_validator/models/detector.py` - PyTorch Lightning modules for 3D ResNet with attention
- **Training**: `mri_node_validator/scripts/train.py` - Hugging Face Trainer integration
- **Inference**: `mri_node_validator/scripts/infer.py` - Batch and single-image prediction
- **Evaluation**: `mri_node_validator/evaluation/` - Comprehensive metrics and visualization

### Key Design Patterns

1. **CSV-Based Label Management**: File keys extracted from NIfTI filenames match CSV entries
   - Pattern: `patient_001_t1.nii.gz` → file_key: `patient_001_t1`
   - Configurable via `key_pattern` in YAML configs

2. **YAML Configuration System**: All parameters centralized in config files
   - `configs/training_config.yaml` - Complete training pipeline configuration
   - `configs/inference_config.yaml` - Inference and preprocessing settings

3. **Hugging Face Integration**: Leverages Trainer API for robust training infrastructure
   - Custom model wrapping for 3D medical imaging
   - Automatic checkpointing and metric computation

4. **torchio Preprocessing Pipeline**: Medical image-specific transformations
   - Resampling, intensity normalization, augmentation
   - Configurable preprocessing parameters

### Data Flow

1. **NIfTI files** → torchio Subject creation
2. **CSV labels** → file_key matching for label assignment
3. **Preprocessing** → resampling, normalization, augmentation
4. **DataLoader** → batch creation for training/inference
5. **Model** → 3D ResNet backbone + classification head
6. **Evaluation** → comprehensive metrics and visualization

## Development Notes

### File Structure Logic
- All main code in `mri_node_validator/` package
- Scripts in `scripts/` for training/inference entry points
- Configs in `configs/` with YAML parameter files
- Tests follow pytest conventions with fixtures in `conftest.py`

### Testing Strategy
- Fixtures provide sample 3D volumes and segmentation masks
- Tests cover data loading, model architecture, and training components
- Run from repository root to ensure proper module imports

### Configuration Management
- YAML-based configuration with nested structure
- Command-line arguments override config file values
- Separate configs for training vs inference to prevent parameter conflicts

### Medical Imaging Specifics
- 3D volumes with shape (D, H, W) after channel dimension
- NIfTI format support via torchio
- Spacing-aware resampling for consistent voxel dimensions
- Intensity preprocessing with percentile-based clipping

## Examples Directory

The `examples/` directory contains comprehensive tutorials demonstrating framework usage:

### 01_minimal_usage.py
- **Purpose**: Simplest introduction to loading and using pre-trained models
- **Key Features**: Model loading from HuggingFace Hub, basic inference
- **Use Case**: Quick start for new users

### 02_training_basic.py
- **Purpose**: Complete training pipeline from data creation to model saving
- **Key Features**: CSV label setup, dataset creation, training with HuggingFace Trainer
- **Use Case**: Training your first 3D ResNet model

### 03_inference_detailed.py
- **Purpose**: Comprehensive inference with evaluation and visualization
- **Key Features**: Batch processing, evaluation metrics, confusion matrices, ROC curves
- **Use Case**: Evaluating trained models on test datasets

### 04_custom_augmentation.py
- **Purpose**: Advanced training with medical image augmentation
- **Key Features**: TorchIO augmentation pipeline, multiple augmentation strategies, performance comparison
- **Augmentations**: Rotation, flipping, noise, blur, elastic deformation, intensity changes
- **Use Case**: Improving model robustness with data augmentation

### 05_huggingface_hub_integration.py
- **Purpose**: Model sharing and community integration
- **Key Features**: Hub authentication, model card creation, upload/download workflows
- **Use Case**: Publishing and sharing trained models with the community

## Best Practices

### Model Development
- Use the example scripts as starting points for your own projects
- Follow the CSV-based labeling system for consistency
- Implement comprehensive evaluation using the ModelEvaluator class
- Apply appropriate medical image augmentation for better generalization

### Testing and Quality
- Run all tests before committing changes: `python -m pytest mri_node_validator/tests/`
- Use code formatting: `black mri_node_validator/`
- Perform type checking: `mypy mri_node_validator/`
- Apply linting: `pylint mri_node_validator/`

### Model Sharing
- Create comprehensive model cards with performance metrics
- Follow medical AI ethics guidelines when sharing models
- Include proper documentation and usage examples
- Test model loading and inference before publishing