"""Tests for data collators."""

import pytest
import torch
from typing import List, Dict, Any

from mri_node_validator.data.collator import MRIDataCollator, FilteredMRIDataCollator


class TestMRIDataCollator:
    """Test cases for MRIDataCollator class."""
    
    def test_collator_initialization(self):
        """Test collator initialization."""
        collator = MRIDataCollator()
        assert collator.return_tensors == "pt"
        
        collator_custom = MRIDataCollator(return_tensors="np")
        assert collator_custom.return_tensors == "np"
    
    def test_collate_basic_batch(self, sample_dataset_items):
        """Test basic batch collation."""
        collator = MRIDataCollator()
        
        batch = collator(sample_dataset_items)
        
        # Check that all required keys are present
        assert 'image' in batch
        assert 'segmentation' in batch
        assert 'labels' in batch
        assert 'file_key' in batch
        assert 'file_path' in batch
    
    def test_collate_tensor_stacking(self, sample_dataset_items):
        """Test that tensors are properly stacked."""
        collator = MRIDataCollator()
        
        batch = collator(sample_dataset_items)
        
        batch_size = len(sample_dataset_items)
        
        # Check tensor shapes
        assert batch['image'].shape[0] == batch_size
        assert batch['segmentation'].shape[0] == batch_size
        assert batch['labels'].shape[0] == batch_size
        
        # Check that 3D structure is preserved
        assert batch['image'].dim() == 5  # (B, C, D, H, W)
        assert batch['segmentation'].dim() == 5  # (B, C, D, H, W)
        assert batch['labels'].dim() == 1  # (B,)
    
    def test_collate_metadata_preservation(self, sample_dataset_items):
        """Test that metadata is preserved as lists."""
        collator = MRIDataCollator()
        
        batch = collator(sample_dataset_items)
        
        # Metadata should be lists
        assert isinstance(batch['file_key'], list)
        assert isinstance(batch['file_path'], list)
        
        # Check list lengths
        assert len(batch['file_key']) == len(sample_dataset_items)
        assert len(batch['file_path']) == len(sample_dataset_items)
        
        # Check content preservation
        for i, item in enumerate(sample_dataset_items):
            assert batch['file_key'][i] == item['file_key']
            assert batch['file_path'][i] == item['file_path']
    
    def test_collate_without_segmentation(self):
        """Test collation when segmentation is missing."""
        items = [
            {
                'image': torch.randn(1, 32, 32, 32),
                'labels': torch.tensor(0),
                'file_key': 'test1',
                'file_path': '/path/to/test1'
            },
            {
                'image': torch.randn(1, 32, 32, 32),
                'labels': torch.tensor(1),
                'file_key': 'test2',
                'file_path': '/path/to/test2'
            }
        ]
        
        collator = MRIDataCollator()
        batch = collator(items)
        
        # Should use image as segmentation fallback
        assert 'segmentation' in batch
        assert torch.equal(batch['image'], batch['segmentation'])
    
    def test_collate_empty_batch(self):
        """Test collation with empty input."""
        collator = MRIDataCollator()
        
        batch = collator([])
        
        assert batch == {}
    
    def test_collate_single_item(self):
        """Test collation with single item."""
        item = {
            'image': torch.randn(1, 32, 32, 32),
            'segmentation': torch.randn(1, 32, 32, 32),
            'labels': torch.tensor(1),
            'file_key': 'single_test',
            'file_path': '/path/to/single_test'
        }
        
        collator = MRIDataCollator()
        batch = collator([item])
        
        # Should have batch dimension of 1
        assert batch['image'].shape[0] == 1
        assert batch['segmentation'].shape[0] == 1
        assert batch['labels'].shape[0] == 1
        assert len(batch['file_key']) == 1
        assert len(batch['file_path']) == 1
    
    def test_collate_mixed_tensor_sizes(self):
        """Test collation with different tensor sizes."""
        items = [
            {
                'image': torch.randn(1, 32, 32, 32),
                'segmentation': torch.randn(1, 32, 32, 32),
                'labels': torch.tensor(0),
                'file_key': 'test1',
                'file_path': '/path/to/test1'
            },
            {
                'image': torch.randn(1, 64, 64, 64),  # Different size
                'segmentation': torch.randn(1, 64, 64, 64),
                'labels': torch.tensor(1),
                'file_key': 'test2',
                'file_path': '/path/to/test2'
            }
        ]
        
        collator = MRIDataCollator()
        
        # Should raise an error for mismatched tensor sizes
        with pytest.raises(RuntimeError):
            batch = collator(items)


class TestFilteredMRIDataCollator:
    """Test cases for FilteredMRIDataCollator class."""
    
    def test_filtered_collator_initialization(self):
        """Test filtered collator initialization."""
        collator = FilteredMRIDataCollator()
        
        # Should have default model input keys
        expected_keys = {'image', 'segmentation', 'labels'}
        assert collator.model_input_keys == expected_keys
    
    def test_filtered_collator_custom_keys(self):
        """Test filtered collator with custom input keys."""
        custom_keys = ['image', 'labels']
        collator = FilteredMRIDataCollator(model_input_keys=custom_keys)
        
        assert collator.model_input_keys == set(custom_keys)
    
    def test_filtered_collate_removes_metadata(self, sample_dataset_items):
        """Test that filtered collator removes metadata."""
        collator = FilteredMRIDataCollator()
        
        batch = collator(sample_dataset_items)
        
        # Should only have model input keys
        expected_keys = {'image', 'segmentation', 'labels'}
        assert set(batch.keys()) == expected_keys
        
        # Should not have metadata
        assert 'file_key' not in batch
        assert 'file_path' not in batch
    
    def test_filtered_collate_custom_filtering(self, sample_dataset_items):
        """Test filtered collator with custom filtering."""
        # Only keep image and labels
        collator = FilteredMRIDataCollator(model_input_keys=['image', 'labels'])
        
        batch = collator(sample_dataset_items)
        
        # Should only have specified keys
        expected_keys = {'image', 'labels'}
        assert set(batch.keys()) == expected_keys
        
        # Should not have segmentation or metadata
        assert 'segmentation' not in batch
        assert 'file_key' not in batch
        assert 'file_path' not in batch
    
    def test_filtered_collate_preserves_tensor_properties(self, sample_dataset_items):
        """Test that filtered collator preserves tensor properties."""
        collator = FilteredMRIDataCollator()
        
        batch = collator(sample_dataset_items)
        
        batch_size = len(sample_dataset_items)
        
        # Check tensor shapes are preserved
        assert batch['image'].shape[0] == batch_size
        assert batch['segmentation'].shape[0] == batch_size
        assert batch['labels'].shape[0] == batch_size
        
        # Check dimensions
        assert batch['image'].dim() == 5  # (B, C, D, H, W)
        assert batch['segmentation'].dim() == 5  # (B, C, D, H, W)
        assert batch['labels'].dim() == 1  # (B,)
    
    def test_filtered_collate_empty_keys(self, sample_dataset_items):
        """Test filtered collator with empty model input keys."""
        collator = FilteredMRIDataCollator(model_input_keys=[])
        
        batch = collator(sample_dataset_items)
        
        # Should return empty dict (no keys to keep)
        assert batch == {}
    
    def test_filtered_collate_nonexistent_keys(self, sample_dataset_items):
        """Test filtered collator with nonexistent keys."""
        # Request keys that don't exist in the data
        collator = FilteredMRIDataCollator(model_input_keys=['nonexistent_key'])
        
        batch = collator(sample_dataset_items)
        
        # Should return empty dict (no matching keys)
        assert batch == {}


class TestCollatorIntegration:
    """Integration tests for collators."""
    
    def test_collator_with_dataloader(self, sample_dataset_items):
        """Test collator integration with DataLoader."""
        from torch.utils.data import DataLoader, Dataset
        
        class MockDataset(Dataset):
            def __init__(self, items):
                self.items = items
            
            def __len__(self):
                return len(self.items)
            
            def __getitem__(self, idx):
                return self.items[idx]
        
        dataset = MockDataset(sample_dataset_items)
        collator = MRIDataCollator()
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collator,
            shuffle=False
        )
        
        # Test that we can iterate through the dataloader
        for batch in dataloader:
            assert isinstance(batch, dict)
            assert 'image' in batch
            assert 'labels' in batch
            
            # Check batch size
            assert batch['image'].shape[0] <= 2
            break
    
    def test_filtered_collator_with_dataloader(self, sample_dataset_items):
        """Test filtered collator integration with DataLoader."""
        from torch.utils.data import DataLoader, Dataset
        
        class MockDataset(Dataset):
            def __init__(self, items):
                self.items = items
            
            def __len__(self):
                return len(self.items)
            
            def __getitem__(self, idx):
                return self.items[idx]
        
        dataset = MockDataset(sample_dataset_items)
        collator = FilteredMRIDataCollator(model_input_keys=['image', 'labels'])
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collator,
            shuffle=False
        )
        
        # Test that we can iterate through the dataloader
        for batch in dataloader:
            assert isinstance(batch, dict)
            
            # Should only have filtered keys
            expected_keys = {'image', 'labels'}
            assert set(batch.keys()) == expected_keys
            
            # Should not have metadata
            assert 'file_key' not in batch
            assert 'file_path' not in batch
            break
    
    def test_collator_error_handling(self):
        """Test collator error handling with malformed data."""
        collator = MRIDataCollator()
        
        # Test with None input
        batch = collator(None)
        assert batch == {}
        
        # Test with malformed items
        malformed_items = [
            {'image': "not_a_tensor"},  # Invalid tensor
            {'labels': torch.tensor(0)}   # Missing image
        ]
        
        # Should handle gracefully or raise appropriate error
        try:
            batch = collator(malformed_items)
        except (TypeError, AttributeError, KeyError):
            # Expected to fail with these error types
            pass