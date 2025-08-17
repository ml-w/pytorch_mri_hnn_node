import pytest
import torch
from mri_node_validator.data.loader import DataLoader
from .conftest import sample_mri_volume, sample_segmentation_mask, merged_node_mask, sample_bounding_boxes

class TestDataLoader:
    def test_load_mri_volume(self, sample_mri_volume):
        """Test loading of MRI volumes"""
        loader = DataLoader()
        processed = loader.preprocess_volume(sample_mri_volume)
        assert processed.shape == sample_mri_volume.shape
        assert torch.is_tensor(processed)

    def test_load_segmentation_mask(self, sample_segmentation_mask):
        """Test loading of segmentation masks"""
        loader = DataLoader()
        processed = loader.preprocess_mask(sample_segmentation_mask)
        assert processed.shape == sample_segmentation_mask.shape
        assert torch.is_tensor(processed)

    def test_detect_merged_nodes(self, sample_segmentation_mask, merged_node_mask):
        """Test detection of merged nodes in segmentation masks"""
        loader = DataLoader()
        
        # Test with properly separated nodes
        result = loader.detect_merged_nodes(sample_segmentation_mask)
        assert not result["has_merged_nodes"]
        
        # Test with merged nodes
        result = loader.detect_merged_nodes(merged_node_mask)
        assert result["has_merged_nodes"]
        assert len(result["merged_regions"]) > 0

    def test_load_bounding_boxes(self, sample_bounding_boxes):
        """Test loading of bounding box annotations"""
        loader = DataLoader()
        boxes = loader.process_bounding_boxes(sample_bounding_boxes)
        assert len(boxes) == len(sample_bounding_boxes)
        assert all(isinstance(box, dict) for box in boxes)

    def test_data_augmentation(self, sample_mri_volume, sample_segmentation_mask):
        """Test data augmentation pipeline"""
        loader = DataLoader()
        augmented = loader.augment_data(sample_mri_volume, sample_segmentation_mask)
        assert augmented["image"].shape == sample_mri_volume.shape
        assert augmented["mask"].shape == sample_segmentation_mask.shape