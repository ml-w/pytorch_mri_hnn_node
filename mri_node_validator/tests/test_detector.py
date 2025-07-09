import pytest
import torch
from mri_node_validator.models.detector import NodeMergingDetector
from .conftest import sample_mri_volume, sample_segmentation_mask, merged_node_mask

class TestNodeMergingDetector:
    @pytest.fixture
    def detector(self):
        """Fixture providing initialized detector model"""
        return NodeMergingDetector()

    def test_model_initialization(self, detector):
        """Test model architecture initialization"""
        assert detector is not None
        assert hasattr(detector, 'feature_extractor')
        assert hasattr(detector, 'attention_module')
        assert hasattr(detector, 'classifier')

    def test_forward_pass(self, detector, sample_mri_volume, sample_segmentation_mask):
        """Test forward pass with valid inputs"""
        # Create batch dimension
        volume = sample_mri_volume.unsqueeze(0)
        mask = sample_segmentation_mask.unsqueeze(0)
        
        output = detector(volume, mask)
        assert output.shape == (1, 2)  # batch_size x num_classes
        assert torch.all(output >= 0)  # probabilities should be non-negative

    def test_detect_merged_nodes(self, detector, merged_node_mask):
        """Test merged node detection functionality"""
        # Create batch dimension for merged mask
        merged_mask = merged_node_mask.unsqueeze(0)
        
        # Test with dummy volume (not used in this test)
        dummy_volume = torch.zeros_like(merged_mask)
        
        output = detector(detect_only=True, volume=dummy_volume, mask=merged_mask)
        assert output['has_merged_nodes']
        assert len(output['merged_regions']) > 0

    def test_training_step(self, detector, sample_mri_volume, sample_segmentation_mask):
        """Test training step with mock data"""
        batch = {
            'image': sample_mri_volume.unsqueeze(0),
            'mask': sample_segmentation_mask.unsqueeze(0),
            'label': torch.tensor([0])  # 0 = no merged nodes
        }
        
        loss = detector.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad  # Should be part of computation graph

    def test_validation_step(self, detector, sample_mri_volume, sample_segmentation_mask):
        """Test validation step with mock data"""
        batch = {
            'image': sample_mri_volume.unsqueeze(0),
            'mask': sample_segmentation_mask.unsqueeze(0),
            'label': torch.tensor([0])  # 0 = no merged nodes
        }
        
        metrics = detector.validation_step(batch, batch_idx=0)
        assert 'val_loss' in metrics
        assert 'val_acc' in metrics