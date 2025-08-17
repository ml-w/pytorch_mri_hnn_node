"""Tests for MRI dataset and data module components."""

import pytest
import torch
import pandas as pd
from pathlib import Path
import tempfile
import numpy as np

from mri_node_validator.data.dataset import MRIDataset, MRIDataModule, TransformWrapper
from mri_node_validator.data.transforms import get_training_transforms


class TestMRIDataset:
    """Test cases for MRIDataset class."""
    
    def test_initialization(self, temp_dir, sample_csv_file, sample_nifti_files):
        """Test dataset initialization with valid inputs."""
        dataset = MRIDataset(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            transforms=None
        )
        
        assert len(dataset) > 0
        assert dataset.data_dir == temp_dir
        assert dataset.csv_path == sample_csv_file
        assert len(dataset.file_paths) == len(dataset.file_keys)
        assert len(dataset.file_keys) == len(dataset.labels)
    
    def test_getitem_returns_correct_format(self, temp_dir, sample_csv_file, sample_nifti_files):
        """Test that __getitem__ returns correct dictionary format."""
        dataset = MRIDataset(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            transforms=None
        )
        
        sample = dataset[0]
        
        # Check required keys
        assert 'image' in sample
        assert 'segmentation' in sample
        assert 'labels' in sample
        assert 'file_key' in sample
        assert 'file_path' in sample
        
        # Check tensor shapes
        assert sample['image'].dim() == 4  # (C, D, H, W)
        assert sample['segmentation'].dim() == 4  # (C, D, H, W)
        assert sample['labels'].dim() == 0  # scalar tensor
        
        # Check data types
        assert torch.is_tensor(sample['image'])
        assert torch.is_tensor(sample['segmentation'])
        assert torch.is_tensor(sample['labels'])
        assert isinstance(sample['file_key'], str)
        assert isinstance(sample['file_path'], str)
    
    def test_csv_key_matching(self, temp_dir, sample_nifti_files):
        """Test CSV key matching with different patterns."""
        # Create CSV with specific keys
        csv_data = pd.DataFrame({
            'file_key': ['patient_001_t1', 'patient_002_t1'],
            'label': [0, 1]
        })
        csv_path = temp_dir / "test_labels.csv"
        csv_data.to_csv(csv_path, index=False)
        
        dataset = MRIDataset(
            data_dir=temp_dir,
            csv_path=csv_path,
            key_pattern=r"(.+)\.nii(?:\.gz)?$"
        )
        
        # Should match 2 out of 3 files
        assert len(dataset) == 2
        assert 'patient_001_t1' in dataset.file_keys
        assert 'patient_002_t1' in dataset.file_keys
    
    def test_class_distribution(self, temp_dir, sample_csv_file, sample_nifti_files):
        """Test class distribution computation."""
        dataset = MRIDataset(
            data_dir=temp_dir,
            csv_path=sample_csv_file
        )
        
        distribution = dataset.get_class_distribution()
        
        assert isinstance(distribution, dict)
        assert all(isinstance(k, int) for k in distribution.keys())
        assert all(isinstance(v, int) for v in distribution.values())
        assert sum(distribution.values()) == len(dataset)
    
    def test_get_sample_by_key(self, temp_dir, sample_csv_file, sample_nifti_files):
        """Test retrieving samples by file key."""
        dataset = MRIDataset(
            data_dir=temp_dir,
            csv_path=sample_csv_file
        )
        
        # Test valid key
        if len(dataset) > 0:
            key = dataset.file_keys[0]
            sample = dataset.get_sample_by_key(key)
            assert sample is not None
            assert sample['file_key'] == key
        
        # Test invalid key
        invalid_sample = dataset.get_sample_by_key('nonexistent_key')
        assert invalid_sample is None
    
    def test_transforms_application(self, temp_dir, sample_csv_file, sample_nifti_files):
        """Test that transforms are applied correctly."""
        transforms = get_training_transforms(
            target_spacing=(2.0, 2.0, 2.0),
            target_shape=(32, 32, 32)
        )
        
        dataset = MRIDataset(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            transforms=transforms
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            # Check that shape matches target after transforms
            assert sample['image'].shape[1:] == (32, 32, 32)  # D, H, W
    
    def test_empty_directory(self, temp_dir):
        """Test behavior with empty data directory."""
        # Create empty CSV
        empty_csv = pd.DataFrame({'file_key': [], 'label': []})
        csv_path = temp_dir / "empty.csv"
        empty_csv.to_csv(csv_path, index=False)
        
        dataset = MRIDataset(
            data_dir=temp_dir,
            csv_path=csv_path
        )
        
        assert len(dataset) == 0
    
    def test_missing_csv_columns(self, temp_dir, sample_nifti_files):
        """Test error handling for missing CSV columns."""
        # Create CSV without required columns
        bad_csv = pd.DataFrame({'wrong_column': ['value1', 'value2']})
        csv_path = temp_dir / "bad.csv"
        bad_csv.to_csv(csv_path, index=False)
        
        with pytest.raises(KeyError):
            dataset = MRIDataset(
                data_dir=temp_dir,
                csv_path=csv_path
            )


class TestMRIDataModule:
    """Test cases for MRIDataModule class."""
    
    def test_initialization(self, temp_dir, sample_csv_file):
        """Test data module initialization."""
        data_module = MRIDataModule(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            batch_size=2,
            num_workers=0
        )
        
        assert data_module.data_dir == temp_dir
        assert data_module.csv_path == sample_csv_file
        assert data_module.batch_size == 2
        assert data_module.num_workers == 0
    
    def test_setup_splits(self, temp_dir, sample_csv_file, sample_nifti_files):
        """Test dataset splitting."""
        data_module = MRIDataModule(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            train_split=0.6,
            val_split=0.2,
            test_split=0.2,
            random_seed=42
        )
        
        train_dataset, val_dataset, test_dataset = data_module.setup()
        
        total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
        assert total_samples > 0
        
        # Check that splits are approximately correct
        if total_samples >= 3:  # Only test if we have enough samples
            train_ratio = len(train_dataset) / total_samples
            val_ratio = len(val_dataset) / total_samples
            test_ratio = len(test_dataset) / total_samples
            
            # For small datasets (< 10), allow more variance due to integer rounding
            if total_samples < 10:
                assert 0.2 <= train_ratio <= 0.8  # More lenient for small datasets
                assert 0.0 <= val_ratio <= 0.8    # Very lenient for small datasets
                assert 0.0 <= test_ratio <= 0.8   # Very lenient for small datasets
            else:
                assert 0.4 <= train_ratio <= 0.8  # Stricter for larger datasets
                assert 0.0 <= val_ratio <= 0.4
                assert 0.0 <= test_ratio <= 0.4
    
    def test_get_dataloaders(self, temp_dir, sample_csv_file, sample_nifti_files):
        """Test DataLoader creation."""
        data_module = MRIDataModule(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            batch_size=1,
            num_workers=0
        )
        
        train_loader, val_loader, test_loader = data_module.get_dataloaders()
        
        # Test that loaders are created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Test batch iteration if we have data
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            assert isinstance(batch, dict)
            assert 'image' in batch
            assert 'labels' in batch
    
    def test_invalid_splits(self, temp_dir, sample_csv_file):
        """Test error handling for invalid split ratios."""
        with pytest.raises(ValueError):
            MRIDataModule(
                data_dir=temp_dir,
                csv_path=sample_csv_file,
                train_split=0.5,
                val_split=0.3,
                test_split=0.3  # Sum = 1.1 > 1.0
            )
    
    def test_reproducible_splits(self, temp_dir, sample_csv_file, sample_nifti_files):
        """Test that splits are reproducible with same seed."""
        data_module1 = MRIDataModule(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            random_seed=42
        )
        
        data_module2 = MRIDataModule(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            random_seed=42
        )
        
        train1, val1, test1 = data_module1.setup()
        train2, val2, test2 = data_module2.setup()
        
        # Should have same number of samples in each split
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)
        assert len(test1) == len(test2)


class TestTransformWrapper:
    """Test cases for TransformWrapper class."""
    
    def test_transform_wrapper_initialization(self, sample_dataset_items):
        """Test TransformWrapper initialization."""
        from torch.utils.data import Dataset
        
        class MockDataset(Dataset):
            def __init__(self, items):
                self.items = items
            
            def __len__(self):
                return len(self.items)
            
            def __getitem__(self, idx):
                return self.items[idx]
        
        base_dataset = MockDataset(sample_dataset_items)
        transforms = get_training_transforms(target_shape=(32, 32, 32))
        
        wrapped_dataset = TransformWrapper(base_dataset, transforms)
        
        assert len(wrapped_dataset) == len(base_dataset)
    
    def test_transform_application(self, sample_dataset_items):
        """Test that transforms are applied correctly."""
        from torch.utils.data import Dataset
        
        class MockDataset(Dataset):
            def __init__(self, items):
                self.items = items
            
            def __len__(self):
                return len(self.items)
            
            def __getitem__(self, idx):
                return self.items[idx]
        
        base_dataset = MockDataset(sample_dataset_items)
        transforms = get_training_transforms(target_shape=(32, 32, 32))
        
        wrapped_dataset = TransformWrapper(base_dataset, transforms)
        
        sample = wrapped_dataset[0]
        
        # Check that transforms were applied
        assert sample['image'].shape[1:] == (32, 32, 32)  # D, H, W
        assert sample['segmentation'].shape[1:] == (32, 32, 32)  # D, H, W
        
        # Check that other fields are preserved
        assert 'labels' in sample
        assert 'file_key' in sample
        assert 'file_path' in sample
    
    def test_getitem_preserves_metadata(self, sample_dataset_items):
        """Test that metadata is preserved through transforms."""
        from torch.utils.data import Dataset
        import torchio as tio
        
        class MockDataset(Dataset):
            def __init__(self, items):
                self.items = items
            
            def __len__(self):
                return len(self.items)
            
            def __getitem__(self, idx):
                return self.items[idx]
        
        base_dataset = MockDataset(sample_dataset_items)
        transforms = tio.Compose([tio.ZNormalization()])
        
        wrapped_dataset = TransformWrapper(base_dataset, transforms)
        
        original_sample = base_dataset[0]
        transformed_sample = wrapped_dataset[0]
        
        # Metadata should be identical
        assert transformed_sample['labels'].item() == original_sample['labels'].item()
        assert transformed_sample['file_key'] == original_sample['file_key']
        assert transformed_sample['file_path'] == original_sample['file_path']