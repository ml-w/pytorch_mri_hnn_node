import pytest
import numpy as np
import torch
import tempfile
import pandas as pd
import nibabel as nib
from pathlib import Path
import yaml
import os
from typing import Dict, Any, List


@pytest.fixture
def sample_mri_volume():
    """Fixture providing a sample 3D MRI volume"""
    return torch.randn(1, 64, 64, 64)  # (C, D, H, W)


@pytest.fixture
def sample_mri_volume_large():
    """Fixture providing a larger 3D MRI volume for memory tests"""
    return torch.randn(1, 128, 128, 128)  # (C, D, H, W)


@pytest.fixture
def sample_mri_volume_batch():
    """Fixture providing a batch of 3D MRI volumes"""
    return torch.randn(4, 1, 64, 64, 64)  # (B, C, D, H, W)


@pytest.fixture
def sample_segmentation_mask():
    """Fixture providing a sample segmentation mask"""
    mask = torch.zeros(1, 64, 64, 64)
    mask[:, 10:20, 10:20, 10:20] = 1  # Single node
    mask[:, 30:40, 30:40, 30:40] = 2  # Another node
    return mask


@pytest.fixture
def merged_node_mask():
    """Fixture providing a mask with merged nodes"""
    mask = torch.zeros(1, 64, 64, 64)
    mask[:, 10:30, 10:30, 10:30] = 1  # Merged nodes
    return mask


@pytest.fixture
def sample_bounding_boxes():
    """Fixture providing sample bounding boxes in JSON format"""
    return [
        {"label": 1, "coordinates": [10, 10, 10, 20, 20, 20]},
        {"label": 2, "coordinates": [30, 30, 30, 40, 40, 40]}
    ]


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_csv_data():
    """Fixture providing sample CSV data for testing"""
    return pd.DataFrame({
        'file_key': ['patient_001_t1', 'patient_002_t1', 'patient_003_t2'],
        'label': [0, 1, 0],
        'age': [45, 67, 34],
        'sex': ['M', 'F', 'M']
    })


@pytest.fixture
def sample_nifti_files(temp_dir, sample_mri_volume):
    """Fixture creating sample NIfTI files in temporary directory"""
    files = []
    for i, filename in enumerate(['patient_001_t1.nii.gz', 'patient_002_t1.nii.gz', 'patient_003_t2.nii.gz']):
        # Create realistic NIfTI data
        data = np.random.randn(64, 64, 64).astype(np.float32)
        img = nib.Nifti1Image(data, affine=np.eye(4))
        
        file_path = temp_dir / filename
        nib.save(img, str(file_path))
        files.append(file_path)
    
    return files


@pytest.fixture
def sample_csv_file(temp_dir, sample_csv_data):
    """Fixture creating a sample CSV file"""
    csv_path = temp_dir / "labels.csv"
    sample_csv_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def training_config():
    """Fixture providing sample training configuration"""
    return {
        'model': {
            'learning_rate': 1e-3,
            'model_type': '3d_resnet_attention',
            'num_classes': 2
        },
        'data': {
            'batch_size': 2,
            'num_workers': 0,  # For testing
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'key_pattern': r"(.+)\.nii(?:\.gz)?$"
        },
        'preprocessing': {
            'target_spacing': [1.0, 1.0, 1.0],
            'target_shape': [64, 64, 64],
            'augmentation_probability': 0.5,
            'intensity_clipping': True,
            'clipping_percentiles': [1.0, 99.0]
        },
        'training': {
            'num_train_epochs': 1,
            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'save_steps': 100,
            'eval_steps': 50,
            'logging_steps': 10,
            'save_total_limit': 2
        },
        'seed': 42
    }


@pytest.fixture
def training_config_file(temp_dir, training_config):
    """Fixture creating a training configuration file"""
    config_path = temp_dir / "training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(training_config, f)
    return config_path


@pytest.fixture
def sample_predictions():
    """Fixture providing sample model predictions"""
    return {
        'predictions': np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]]),
        'labels': np.array([0, 1, 0, 1]),
        'probabilities': np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])
    }


@pytest.fixture
def sample_binary_predictions():
    """Fixture providing sample binary model predictions"""
    return {
        'predictions': np.array([0.8, 0.3, 0.9, 0.4]),
        'labels': np.array([1, 0, 1, 0]),
        'probabilities': np.array([0.8, 0.3, 0.9, 0.4])
    }


@pytest.fixture
def sample_confusion_matrix():
    """Fixture providing a sample confusion matrix"""
    return np.array([[45, 5], [3, 47]])


@pytest.fixture
def sample_metrics():
    """Fixture providing sample metrics dictionary"""
    return {
        'accuracy': 0.92,
        'precision': 0.90,
        'recall': 0.94,
        'f1': 0.92,
        'auc': 0.88
    }


@pytest.fixture
def sample_training_history():
    """Fixture providing sample training history"""
    return {
        'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
        'val_loss': [0.7, 0.5, 0.4, 0.35, 0.25],
        'train_accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
        'val_accuracy': [0.65, 0.72, 0.78, 0.82, 0.87]
    }


@pytest.fixture
def mock_hf_model_config():
    """Fixture providing HF model configuration"""
    from mri_node_validator.models.hf_detector import MRIConfig
    return MRIConfig(
        learning_rate=1e-3,
        architecture_type="3d_resnet_attention",
        num_labels=2
    )


@pytest.fixture
def sample_batch_dict():
    """Fixture providing a sample batch in dictionary format"""
    return {
        'image': torch.randn(2, 1, 64, 64, 64),
        'segmentation': torch.randn(2, 1, 64, 64, 64),
        'labels': torch.tensor([0, 1]),
        'file_key': ['patient_001_t1', 'patient_002_t1'],
        'file_path': ['/path/to/patient_001_t1.nii.gz', '/path/to/patient_002_t1.nii.gz']
    }


@pytest.fixture
def device():
    """Fixture providing device for testing"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing test data directory for the session"""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def class_names():
    """Fixture providing class names for classification"""
    return ["No_Merged", "Merged"]


@pytest.fixture
def sample_dataset_items():
    """Fixture providing sample dataset items"""
    return [
        {
            'image': torch.randn(1, 64, 64, 64),
            'segmentation': torch.randn(1, 64, 64, 64),
            'labels': torch.tensor(0),
            'file_key': 'patient_001_t1',
            'file_path': '/path/to/patient_001_t1.nii.gz'
        },
        {
            'image': torch.randn(1, 64, 64, 64),
            'segmentation': torch.randn(1, 64, 64, 64),
            'labels': torch.tensor(1),
            'file_key': 'patient_002_t1',
            'file_path': '/path/to/patient_002_t1.nii.gz'
        }
    ]