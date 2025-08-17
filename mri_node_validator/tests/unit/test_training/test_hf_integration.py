"""Tests for Hugging Face Trainer integration."""

import pytest
import torch
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from transformers import TrainingArguments, EvalPrediction
from mri_node_validator.scripts.train import MRITrainer, compute_metrics, load_config
from mri_node_validator.models.hf_detector import HFNodeDetector, MRIConfig
from mri_node_validator.data.collator import MRIDataCollator


# Shared fixtures for all test classes
@pytest.fixture
def mock_model(mock_hf_model_config):
    """Fixture providing a mock HF model."""
    return HFNodeDetector(mock_hf_model_config)

@pytest.fixture
def training_args(temp_dir):
    """Fixture providing training arguments."""
    return TrainingArguments(
        output_dir=str(temp_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_steps=10,
        eval_steps=10,
        logging_steps=5,
        remove_unused_columns=False,  # Important for our custom data format
    )


class TestMRITrainer:
    """Test cases for MRITrainer class."""
    
    def test_trainer_initialization(self, mock_model, training_args):
        """Test MRITrainer initialization."""
        trainer = MRITrainer(
            model=mock_model,
            args=training_args,
            compute_metrics=compute_metrics,
            data_collator=MRIDataCollator()
        )
        
        assert trainer.model == mock_model
        assert trainer.args == training_args
        assert trainer.compute_metrics == compute_metrics
    
    def test_trainer_compute_loss_basic(self, mock_model, training_args, sample_batch_dict):
        """Test basic loss computation."""
        trainer = MRITrainer(
            model=mock_model,
            args=training_args,
            data_collator=MRIDataCollator()
        )
        
        loss = trainer.compute_loss(mock_model, sample_batch_dict)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0.0
    
    def test_trainer_compute_loss_with_outputs(self, mock_model, training_args, sample_batch_dict):
        """Test loss computation with return_outputs=True."""
        trainer = MRITrainer(
            model=mock_model,
            args=training_args,
            data_collator=MRIDataCollator()
        )
        
        loss, outputs = trainer.compute_loss(mock_model, sample_batch_dict, return_outputs=True)
        
        assert isinstance(loss, torch.Tensor)
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
    
    def test_trainer_input_filtering(self, mock_model, training_args, sample_batch_dict):
        """Test that trainer filters non-model inputs."""
        trainer = MRITrainer(
            model=mock_model,
            args=training_args,
            data_collator=MRIDataCollator()
        )
        
        # Add extra keys that should be filtered
        batch_with_extra = sample_batch_dict.copy()
        batch_with_extra['extra_key'] = 'should_be_filtered'
        
        # Should not raise error due to unexpected keys
        loss = trainer.compute_loss(mock_model, batch_with_extra)
        assert isinstance(loss, torch.Tensor)
    
    def test_trainer_with_dataloader(self, mock_model, training_args, sample_dataset_items):
        """Test trainer with actual dataloader."""
        from torch.utils.data import DataLoader, Dataset
        from mri_node_validator.data.collator import MRIDataCollator
        
        class MockDataset(Dataset):
            def __init__(self, items):
                self.items = items
            
            def __len__(self):
                return len(self.items)
            
            def __getitem__(self, idx):
                return self.items[idx]
        
        dataset = MockDataset(sample_dataset_items)
        collator = MRIDataCollator()
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator)
        
        trainer = MRITrainer(
            model=mock_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
            compute_metrics=compute_metrics
        )
        
        # Test that trainer can handle the dataloader format
        for batch in dataloader:
            loss = trainer.compute_loss(mock_model, batch)
            assert isinstance(loss, torch.Tensor)
            break


class TestComputeMetrics:
    """Test cases for compute_metrics function."""
    
    def test_compute_metrics_binary_classification(self, sample_binary_predictions):
        """Test compute_metrics with binary classification."""
        # Simulate binary model output (single value per sample)
        predictions = sample_binary_predictions['probabilities'].reshape(-1, 1)
        labels = sample_binary_predictions['labels']
        
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_metrics(eval_pred)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        
        # Check value ranges
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0
    
    def test_compute_metrics_binary_1d(self, sample_binary_predictions):
        """Test compute_metrics with 1D binary predictions."""
        # Simulate binary model output (1D array)
        predictions = sample_binary_predictions['probabilities']
        labels = sample_binary_predictions['labels']
        
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_metrics(eval_pred)
        
        assert isinstance(metrics, dict)
        assert all(key in metrics for key in ['accuracy', 'precision', 'recall', 'f1', 'auc'])
    
    def test_compute_metrics_multiclass(self, sample_predictions):
        """Test compute_metrics with multi-class classification."""
        predictions = sample_predictions['predictions']
        labels = sample_predictions['labels']
        
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_metrics(eval_pred)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
    
    def test_compute_metrics_edge_cases(self):
        """Test compute_metrics with edge cases."""
        # All same predictions
        predictions = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        labels = np.array([0, 0, 0])
        
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_metrics(eval_pred)
        
        # Should handle gracefully (perfect accuracy, but undefined precision/recall for other class)
        assert isinstance(metrics, dict)
        assert metrics['accuracy'] == 1.0
    
    def test_compute_metrics_with_errors(self):
        """Test compute_metrics error handling."""
        # Test with mismatched shapes
        predictions = np.array([[1.0, 0.0]])
        labels = np.array([0, 1])  # Different length
        
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        
        # Should handle gracefully or raise appropriate error
        try:
            metrics = compute_metrics(eval_pred)
        except (ValueError, IndexError):
            # Expected to fail with mismatched shapes
            pass


class TestConfigLoading:
    """Test cases for configuration loading."""
    
    def test_load_config_yaml(self, training_config_file):
        """Test loading configuration from YAML file."""
        config = load_config(str(training_config_file))
        
        assert isinstance(config, dict)
        assert 'model' in config
        assert 'data' in config
        assert 'preprocessing' in config
        assert 'training' in config
    
    def test_load_config_nonexistent_file(self):
        """Test error handling for nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent_config.yaml')
    
    def test_load_config_invalid_yaml(self, temp_dir):
        """Test error handling for invalid YAML."""
        invalid_yaml = temp_dir / "invalid.yaml"
        with open(invalid_yaml, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(Exception):  # YAML parsing error
            load_config(str(invalid_yaml))


class TestDataCollatorIntegration:
    """Test cases for data collator integration with HF Trainer."""
    
    def test_collator_with_trainer(self, mock_model, training_args, sample_dataset_items):
        """Test data collator integration with trainer."""
        from torch.utils.data import Dataset
        from mri_node_validator.data.collator import MRIDataCollator
        
        class MockDataset(Dataset):
            def __init__(self, items):
                self.items = items
            
            def __len__(self):
                return len(self.items)
            
            def __getitem__(self, idx):
                return self.items[idx]
        
        dataset = MockDataset(sample_dataset_items)
        collator = MRIDataCollator()
        
        trainer = MRITrainer(
            model=mock_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
            compute_metrics=compute_metrics
        )
        
        # Test data loading
        train_dataloader = trainer.get_train_dataloader()
        for batch in train_dataloader:
            assert isinstance(batch, dict)
            assert 'image' in batch
            assert 'segmentation' in batch
            assert 'labels' in batch
            break
    
    def test_filtered_collator_with_trainer(self, mock_model, training_args, sample_dataset_items):
        """Test filtered data collator integration."""
        from torch.utils.data import Dataset
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
        
        trainer = MRITrainer(
            model=mock_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
            compute_metrics=compute_metrics
        )
        
        # Test data loading with filtered inputs
        train_dataloader = trainer.get_train_dataloader()
        for batch in train_dataloader:
            assert isinstance(batch, dict)
            # Should only have model inputs
            expected_keys = {'image', 'segmentation', 'labels'}
            assert set(batch.keys()) == expected_keys
            break


class TestTrainingArguments:
    """Test cases for training arguments configuration."""
    
    def test_training_arguments_basic(self, temp_dir):
        """Test basic training arguments."""
        args = TrainingArguments(
            output_dir=str(temp_dir),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            remove_unused_columns=False
        )
        
        assert args.output_dir == str(temp_dir)
        assert args.num_train_epochs == 1
        assert args.per_device_train_batch_size == 2
        assert args.remove_unused_columns == False
    
    def test_training_arguments_evaluation(self, temp_dir):
        """Test training arguments with evaluation settings."""
        args = TrainingArguments(
            output_dir=str(temp_dir),
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            logging_steps=10,
            remove_unused_columns=False
        )
        
        assert args.eval_strategy == "steps"
        assert args.eval_steps == 50
        assert args.save_steps == 100
        assert args.logging_steps == 10
    
    def test_training_arguments_with_config(self, training_config):
        """Test training arguments from config."""
        training_config_section = training_config['training']
        
        args = TrainingArguments(
            output_dir="./test_output",
            remove_unused_columns=False,
            **training_config_section
        )
        
        assert args.num_train_epochs == training_config_section['num_train_epochs']
        assert args.per_device_train_batch_size == training_config_section['per_device_train_batch_size']


class TestHFIntegrationEnd2End:
    """End-to-end integration tests for HF Trainer."""
    
    def test_minimal_training_loop(self, mock_model, sample_dataset_items, temp_dir):
        """Test minimal training loop."""
        from torch.utils.data import Dataset
        from mri_node_validator.data.collator import MRIDataCollator
        
        class MockDataset(Dataset):
            def __init__(self, items):
                self.items = items
            
            def __len__(self):
                return len(self.items)
            
            def __getitem__(self, idx):
                return self.items[idx]
        
        dataset = MockDataset(sample_dataset_items)
        collator = MRIDataCollator()
        
        training_args = TrainingArguments(
            output_dir=str(temp_dir),
            max_steps=2,  # Short training for test
            per_device_train_batch_size=1,
            save_strategy="no",  # Disable all saving for test
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        trainer = MRITrainer(
            model=mock_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
            compute_metrics=compute_metrics
        )
        
        # Test training for a few steps
        trainer.train()
        
        # Check that model state changed
        assert mock_model.training
    
    def test_evaluation_loop(self, mock_model, sample_dataset_items, temp_dir):
        """Test evaluation loop."""
        from torch.utils.data import Dataset
        from mri_node_validator.data.collator import MRIDataCollator
        
        class MockDataset(Dataset):
            def __init__(self, items):
                self.items = items
            
            def __len__(self):
                return len(self.items)
            
            def __getitem__(self, idx):
                return self.items[idx]
        
        dataset = MockDataset(sample_dataset_items)
        collator = MRIDataCollator()
        
        training_args = TrainingArguments(
            output_dir=str(temp_dir),
            per_device_eval_batch_size=1,
            remove_unused_columns=False,
            report_to=None,
        )
        
        trainer = MRITrainer(
            model=mock_model,
            args=training_args,
            eval_dataset=dataset,
            data_collator=collator,
            compute_metrics=compute_metrics
        )
        
        # Test evaluation
        eval_results = trainer.evaluate()
        
        assert isinstance(eval_results, dict)
        assert any(key.startswith('eval_') for key in eval_results.keys())
    
    def test_prediction_loop(self, mock_model, sample_dataset_items, temp_dir):
        """Test prediction loop."""
        from torch.utils.data import Dataset
        from mri_node_validator.data.collator import MRIDataCollator
        
        class MockDataset(Dataset):
            def __init__(self, items):
                self.items = items
            
            def __len__(self):
                return len(self.items)
            
            def __getitem__(self, idx):
                return self.items[idx]
        
        dataset = MockDataset(sample_dataset_items)
        collator = MRIDataCollator()
        
        training_args = TrainingArguments(
            output_dir=str(temp_dir),
            per_device_eval_batch_size=1,
            remove_unused_columns=False,
            report_to=None,
        )
        
        trainer = MRITrainer(
            model=mock_model,
            args=training_args,
            data_collator=collator,
            compute_metrics=compute_metrics
        )
        
        # Test prediction
        predictions = trainer.predict(dataset)
        
        assert hasattr(predictions, 'predictions')
        assert hasattr(predictions, 'label_ids')
        assert predictions.predictions.shape[0] == len(dataset)
    
    def test_model_saving_and_loading(self, mock_model, temp_dir):
        """Test model saving and loading with HF format."""
        training_args = TrainingArguments(
            output_dir=str(temp_dir),
            save_strategy="epoch",
            num_train_epochs=1,
            remove_unused_columns=False,
            report_to=None,
        )
        
        trainer = MRITrainer(
            model=mock_model,
            args=training_args,
        )
        
        # Save model
        trainer.save_model()
        
        # Check that files were created
        assert (temp_dir / "config.json").exists()
        # Newer HF versions save as safetensors by default
        assert (temp_dir / "model.safetensors").exists() or (temp_dir / "pytorch_model.bin").exists()
        
        # Load model
        loaded_model = HFNodeDetector.from_pretrained(str(temp_dir))
        
        assert loaded_model.config.num_labels == mock_model.config.num_labels
    
    @patch('mri_node_validator.scripts.train.get_last_checkpoint')
    def test_resume_from_checkpoint(self, mock_get_checkpoint, mock_model, sample_dataset_items, temp_dir):
        """Test resuming training from checkpoint."""
        from torch.utils.data import Dataset
        from mri_node_validator.data.collator import MRIDataCollator
        
        # Mock checkpoint detection
        checkpoint_dir = temp_dir / "checkpoint-100"
        checkpoint_dir.mkdir()
        mock_get_checkpoint.return_value = str(checkpoint_dir)
        
        class MockDataset(Dataset):
            def __init__(self, items):
                self.items = items
            
            def __len__(self):
                return len(self.items)
            
            def __getitem__(self, idx):
                return self.items[idx]
        
        dataset = MockDataset(sample_dataset_items)
        collator = MRIDataCollator()
        
        training_args = TrainingArguments(
            output_dir=str(temp_dir),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            remove_unused_columns=False,
            report_to=None,
        )
        
        trainer = MRITrainer(
            model=mock_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator
        )
        
        # This would normally resume from checkpoint, but we'll just verify the setup
        assert trainer.model is not None
        assert trainer.args.output_dir == str(temp_dir)