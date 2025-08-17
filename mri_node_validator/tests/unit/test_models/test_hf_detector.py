"""Tests for Hugging Face compatible model wrapper."""

import pytest
import torch
import tempfile
from pathlib import Path

from mri_node_validator.models.hf_detector import HFNodeDetector, MRIConfig


class TestMRIConfig:
    """Test cases for MRIConfig class."""
    
    def test_config_initialization_default(self):
        """Test MRIConfig initialization with default values."""
        config = MRIConfig()
        
        assert config.learning_rate == 1e-3
        assert config.architecture_type == "3d_resnet_attention"
        assert config.num_labels == 2
        assert MRIConfig.model_type == "mri_detector"
    
    def test_config_initialization_custom(self):
        """Test MRIConfig initialization with custom values."""
        config = MRIConfig(
            learning_rate=2e-4,
            architecture_type="custom_model",
            num_labels=3
        )
        
        assert config.learning_rate == 2e-4
        assert config.architecture_type == "custom_model"
        assert config.num_labels == 3
    
    def test_config_serialization(self):
        """Test MRIConfig serialization and deserialization."""
        config = MRIConfig(learning_rate=5e-4, num_labels=5)
        
        # Convert to dict and back
        config_dict = config.to_dict()
        new_config = MRIConfig.from_dict(config_dict)
        
        assert new_config.learning_rate == config.learning_rate
        assert new_config.architecture_type == config.architecture_type
        assert new_config.num_labels == config.num_labels
    
    def test_config_json_serialization(self, temp_dir):
        """Test MRIConfig JSON serialization."""
        config = MRIConfig(learning_rate=1e-4, num_labels=4)
        
        # Save to JSON
        json_path = temp_dir / "config.json"
        config.save_pretrained(str(temp_dir))
        
        # Load from JSON
        loaded_config = MRIConfig.from_pretrained(str(temp_dir))
        
        assert loaded_config.learning_rate == config.learning_rate
        assert loaded_config.num_labels == config.num_labels


class TestHFNodeDetector:
    """Test cases for HFNodeDetector class."""
    
    @pytest.fixture
    def sample_config(self):
        """Fixture providing a sample MRI config."""
        return MRIConfig(
            learning_rate=1e-3,
            architecture_type="3d_resnet_attention",
            num_labels=2
        )
    
    def test_model_initialization(self, sample_config):
        """Test HFNodeDetector initialization."""
        model = HFNodeDetector(sample_config)
        
        assert model.config == sample_config
        assert model.num_labels == sample_config.num_labels
        assert hasattr(model, 'detector')
        assert model.detector is not None
    
    def test_model_forward_basic(self, sample_config, sample_mri_volume_batch):
        """Test basic forward pass."""
        model = HFNodeDetector(sample_config)
        batch_size = sample_mri_volume_batch.shape[0]
        
        outputs = model(
            image=sample_mri_volume_batch,
            segmentation=sample_mri_volume_batch
        )
        
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
        assert outputs['logits'].shape[0] == batch_size
        assert outputs['logits'].shape[1] == 1  # Binary classification
    
    def test_model_forward_with_labels(self, sample_config, sample_mri_volume_batch):
        """Test forward pass with labels (training mode)."""
        model = HFNodeDetector(sample_config)
        batch_size = sample_mri_volume_batch.shape[0]
        labels = torch.randint(0, 2, (batch_size,))
        
        outputs = model(
            image=sample_mri_volume_batch,
            segmentation=sample_mri_volume_batch,
            labels=labels
        )
        
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['loss'] is not None
        assert outputs['loss'].requires_grad
    
    def test_model_forward_without_segmentation(self, sample_config, sample_mri_volume_batch):
        """Test forward pass without segmentation (should use image as fallback)."""
        model = HFNodeDetector(sample_config)
        
        outputs = model(image=sample_mri_volume_batch)
        
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
        assert outputs['logits'].shape[0] == sample_mri_volume_batch.shape[0]
    
    def test_model_forward_multiclass(self, sample_mri_volume_batch):
        """Test forward pass with multi-class configuration."""
        config = MRIConfig(num_labels=3)
        model = HFNodeDetector(config)
        batch_size = sample_mri_volume_batch.shape[0]
        labels = torch.randint(0, 3, (batch_size,))
        
        outputs = model(
            image=sample_mri_volume_batch,
            segmentation=sample_mri_volume_batch,
            labels=labels
        )
        
        assert 'loss' in outputs
        assert outputs['loss'] is not None
    
    def test_model_input_filtering(self, sample_config, sample_batch_dict):
        """Test that model handles extra inputs gracefully."""
        model = HFNodeDetector(sample_config)
        
        # Should work even with extra keys
        outputs = model(**sample_batch_dict)
        
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
    
    def test_loss_computation_binary(self, sample_config, sample_mri_volume_batch):
        """Test loss computation for binary classification."""
        model = HFNodeDetector(sample_config)
        batch_size = sample_mri_volume_batch.shape[0]
        labels = torch.randint(0, 2, (batch_size,)).float()
        
        outputs = model(
            image=sample_mri_volume_batch,
            segmentation=sample_mri_volume_batch,
            labels=labels
        )
        
        loss = outputs['loss']
        logits = outputs['logits']
        
        # Check loss properties
        assert loss.item() >= 0.0  # BCE loss should be non-negative
        assert loss.requires_grad
        
        # Check logits shape for binary classification
        assert logits.shape == (batch_size, 1)
    
    def test_get_detector_method(self, sample_config):
        """Test get_detector method."""
        model = HFNodeDetector(sample_config)
        
        detector = model.get_detector()
        
        from mri_node_validator.models.detector import NodeDetector
        assert isinstance(detector, NodeDetector)
    
    def test_model_save_and_load(self, sample_config, temp_dir):
        """Test model saving and loading."""
        model = HFNodeDetector(sample_config)
        
        # Save model with safe_serialization=False to handle duplicate tensor references
        model.save_pretrained(str(temp_dir), safe_serialization=False)
        
        # Check that files were created
        assert (temp_dir / "config.json").exists()
        assert (temp_dir / "pytorch_model.bin").exists()
        
        # Load model
        loaded_model = HFNodeDetector.from_pretrained(str(temp_dir))
        
        assert loaded_model.config.learning_rate == sample_config.learning_rate
        assert loaded_model.config.num_labels == sample_config.num_labels
    
    def test_model_parameters(self, sample_config):
        """Test that model has trainable parameters."""
        model = HFNodeDetector(sample_config)
        
        params = list(model.parameters())
        assert len(params) > 0
        
        # Check that parameters require gradients
        trainable_params = [p for p in params if p.requires_grad]
        assert len(trainable_params) > 0
    
    def test_model_eval_mode(self, sample_config, sample_mri_volume_batch):
        """Test model in evaluation mode."""
        model = HFNodeDetector(sample_config)
        model.eval()
        
        with torch.no_grad():
            outputs = model(
                image=sample_mri_volume_batch,
                segmentation=sample_mri_volume_batch
            )
        
        assert 'logits' in outputs
        # In eval mode, should not have loss without labels
        assert 'loss' not in outputs or outputs['loss'] is None
    
    def test_model_device_compatibility(self, sample_config, device):
        """Test model device compatibility."""
        model = HFNodeDetector(sample_config)
        model = model.to(device)
        
        # Create input on same device
        batch_size = 2
        image = torch.randn(batch_size, 1, 32, 32, 32, device=device)
        segmentation = torch.randn(batch_size, 1, 32, 32, 32, device=device)
        
        outputs = model(image=image, segmentation=segmentation)
        
        # Output should be on same device
        assert outputs['logits'].device == device
    
    def test_model_gradient_flow(self, sample_config, sample_mri_volume_batch):
        """Test gradient flow through model."""
        model = HFNodeDetector(sample_config)
        batch_size = sample_mri_volume_batch.shape[0]
        labels = torch.randint(0, 2, (batch_size,)).float()
        
        # Enable gradients
        sample_mri_volume_batch.requires_grad_(True)
        
        outputs = model(
            image=sample_mri_volume_batch,
            segmentation=sample_mri_volume_batch,
            labels=labels
        )
        
        loss = outputs['loss']
        loss.backward()
        
        # Check that gradients exist
        assert sample_mri_volume_batch.grad is not None
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_output_range(self, sample_config, sample_mri_volume_batch):
        """Test that model outputs are in expected range."""
        model = HFNodeDetector(sample_config)
        
        outputs = model(
            image=sample_mri_volume_batch,
            segmentation=sample_mri_volume_batch
        )
        
        logits = outputs['logits']
        
        # For binary classification with sigmoid, outputs should be in [0, 1]
        probs = torch.sigmoid(logits)
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)


class TestHFDetectorIntegration:
    """Integration tests for HFNodeDetector."""
    
    def test_integration_with_dataloader(self, mock_hf_model_config, sample_dataset_items):
        """Test HFNodeDetector integration with data loader."""
        from torch.utils.data import DataLoader, Dataset
        from mri_node_validator.data.collator import FilteredMRIDataCollator
        
        class MockDataset(Dataset):
            def __init__(self, items):
                self.items = items
            
            def __len__(self):
                return len(self.items)
            
            def __getitem__(self, idx):
                return self.items[idx]
        
        dataset = MockDataset(sample_dataset_items)
        collator = FilteredMRIDataCollator()
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)
        
        model = HFNodeDetector(mock_hf_model_config)
        model.eval()
        
        # Test inference on batch
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(**batch)
                assert 'logits' in outputs
                break
    
    def test_model_config_compatibility(self):
        """Test compatibility between different model configurations."""
        configs = [
            MRIConfig(num_labels=2, learning_rate=1e-3),
            MRIConfig(num_labels=2, learning_rate=5e-4),
            MRIConfig(num_labels=3, learning_rate=1e-3),
        ]
        
        for config in configs:
            model = HFNodeDetector(config)
            model.eval()  # Set to eval mode to avoid BatchNorm issues with small inputs
            
            # Test basic functionality
            batch_size = 1
            image = torch.randn(batch_size, 1, 32, 32, 32)
            
            outputs = model(image=image)
            assert 'logits' in outputs
            
            # Check output shape matches configuration
            expected_output_dim = 1 if config.num_labels == 2 else config.num_labels
            assert outputs['logits'].shape[1] == expected_output_dim
    
    def test_model_memory_efficiency(self, mock_hf_model_config):
        """Test model memory usage with large inputs."""
        model = HFNodeDetector(mock_hf_model_config)
        
        model.eval()  # Set to eval mode
        
        # Test with larger volume (but still reasonable for testing)
        large_volume = torch.randn(1, 1, 64, 64, 64)
        
        with torch.no_grad():
            outputs = model(image=large_volume)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape[0] == 1
    
    def test_model_reproducibility(self, mock_hf_model_config, sample_mri_volume_batch):
        """Test model output reproducibility."""
        torch.manual_seed(42)
        model1 = HFNodeDetector(mock_hf_model_config)
        model1.eval()
        
        torch.manual_seed(42)
        model2 = HFNodeDetector(mock_hf_model_config)
        model2.eval()
        
        # Models should have same weights
        with torch.no_grad():
            outputs1 = model1(image=sample_mri_volume_batch)
            outputs2 = model2(image=sample_mri_volume_batch)
        
        assert torch.allclose(outputs1['logits'], outputs2['logits'], atol=1e-6)