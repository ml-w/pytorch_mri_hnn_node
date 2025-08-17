"""Tests for evaluation components."""

import pytest
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mri_node_validator.evaluation.evaluator import Evaluator, ModelEvaluator
from mri_node_validator.evaluation.metrics import ClassificationMetrics
from mri_node_validator.models.hf_detector import HFNodeDetector, MRIConfig


# Shared fixtures for evaluator tests
@pytest.fixture
def mock_model(mock_hf_model_config):
    """Fixture providing a mock HF model."""
    return HFNodeDetector(mock_hf_model_config)


class TestClassificationMetrics:
    """Test cases for ClassificationMetrics class."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ClassificationMetrics()
        
        assert metrics.metrics is not None
        assert hasattr(metrics, 'metrics')
    
    def test_metrics_update(self, sample_binary_predictions):
        """Test metrics update functionality."""
        metrics = ClassificationMetrics()
        
        preds = torch.tensor(sample_binary_predictions['probabilities'])
        targets = torch.tensor(sample_binary_predictions['labels'])
        
        metrics.update(preds, targets)
        
        # Should not raise any errors
        assert True
    
    def test_metrics_compute(self, sample_binary_predictions):
        """Test metrics computation."""
        metrics = ClassificationMetrics()
        
        preds = torch.tensor(sample_binary_predictions['probabilities'])
        targets = torch.tensor(sample_binary_predictions['labels'])
        
        metrics.update(preds, targets)
        results = metrics.compute()
        
        assert isinstance(results, dict)
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 'auroc' in results
        
        # Check value ranges
        for key, value in results.items():
            assert 0.0 <= value.item() <= 1.0
    
    def test_metrics_generate_report(self, sample_binary_predictions):
        """Test metrics report generation."""
        metrics = ClassificationMetrics()
        
        preds = torch.tensor(sample_binary_predictions['probabilities'])
        targets = torch.tensor(sample_binary_predictions['labels'])
        
        metrics.update(preds, targets)
        report = metrics.generate_report()
        
        assert isinstance(report, str)
        assert "Accuracy:" in report
        assert "Precision:" in report
        assert "Recall:" in report
        assert "F1 Score:" in report
        assert "AUROC:" in report
    
    def test_metrics_generate_report_with_save(self, sample_binary_predictions, temp_dir):
        """Test metrics report generation with save."""
        metrics = ClassificationMetrics()
        
        preds = torch.tensor(sample_binary_predictions['probabilities'])
        targets = torch.tensor(sample_binary_predictions['labels'])
        
        metrics.update(preds, targets)
        
        save_path = temp_dir / "metrics_report.txt"
        report = metrics.generate_report(save_path=str(save_path))
        
        assert isinstance(report, str)
        assert save_path.exists()
        
        # Check file content
        with open(save_path, 'r') as f:
            content = f.read()
            assert content == report
    
    def test_metrics_get_dict(self, sample_binary_predictions):
        """Test getting metrics as dictionary."""
        metrics = ClassificationMetrics()
        
        preds = torch.tensor(sample_binary_predictions['probabilities'])
        targets = torch.tensor(sample_binary_predictions['labels'])
        
        metrics.update(preds, targets)
        metrics_dict = metrics.get_metrics_dict()
        
        assert isinstance(metrics_dict, dict)
        assert all(isinstance(v, float) for v in metrics_dict.values())


class TestEvaluator:
    """Test cases for Evaluator class."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = Evaluator()
        
        assert evaluator.num_classes == 2
        assert evaluator.class_names == ["Class_0", "Class_1"]
        assert evaluator.average == 'weighted'
        assert len(evaluator.all_predictions) == 0
        assert len(evaluator.all_targets) == 0
        assert len(evaluator.all_probabilities) == 0
    
    def test_evaluator_custom_initialization(self, class_names):
        """Test evaluator initialization with custom parameters."""
        evaluator = Evaluator(
            num_classes=2,
            class_names=class_names,
            average='macro'
        )
        
        assert evaluator.num_classes == 2
        assert evaluator.class_names == class_names
        assert evaluator.average == 'macro'
    
    def test_evaluator_update_numpy(self, sample_predictions):
        """Test evaluator update with numpy arrays."""
        evaluator = Evaluator()
        
        predictions = np.argmax(sample_predictions['predictions'], axis=1)
        targets = sample_predictions['labels']
        probabilities = sample_predictions['probabilities']
        
        evaluator.update(predictions, targets, probabilities)
        
        assert len(evaluator.all_predictions) == len(predictions)
        assert len(evaluator.all_targets) == len(targets)
        assert len(evaluator.all_probabilities) == len(probabilities)
    
    def test_evaluator_update_torch(self, sample_predictions):
        """Test evaluator update with torch tensors."""
        evaluator = Evaluator()
        
        predictions = torch.tensor(np.argmax(sample_predictions['predictions'], axis=1))
        targets = torch.tensor(sample_predictions['labels'])
        probabilities = torch.tensor(sample_predictions['probabilities'])
        
        evaluator.update(predictions, targets, probabilities)
        
        assert len(evaluator.all_predictions) == len(predictions)
        assert len(evaluator.all_targets) == len(targets)
        assert len(evaluator.all_probabilities) == len(probabilities)
    
    def test_evaluator_update_binary(self, sample_binary_predictions):
        """Test evaluator update with binary predictions."""
        evaluator = Evaluator()
        
        predictions = (sample_binary_predictions['probabilities'] > 0.5).astype(int)
        targets = sample_binary_predictions['labels']
        probabilities = sample_binary_predictions['probabilities']
        
        evaluator.update(predictions, targets, probabilities)
        
        assert len(evaluator.all_predictions) == len(predictions)
        assert len(evaluator.all_targets) == len(targets)
    
    def test_evaluator_compute_metrics(self, sample_predictions):
        """Test metrics computation."""
        evaluator = Evaluator()
        
        predictions = np.argmax(sample_predictions['predictions'], axis=1)
        targets = sample_predictions['labels']
        probabilities = sample_predictions['probabilities']
        
        evaluator.update(predictions, targets, probabilities)
        metrics = evaluator.compute_metrics()
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Validate against sklearn
        expected_accuracy = accuracy_score(targets, predictions)
        assert abs(metrics['accuracy'] - expected_accuracy) < 1e-6
    
    def test_evaluator_compute_metrics_multiclass(self, sample_predictions):
        """Test metrics computation for multi-class."""
        # Create a 3-class scenario for true multiclass testing
        multiclass_names = ["Class_A", "Class_B", "Class_C"]
        evaluator = Evaluator(num_classes=3, class_names=multiclass_names)
        
        # Modify predictions to be 3-class
        predictions = np.array([0, 1, 2, 1])  # 3-class predictions
        targets = np.array([0, 1, 2, 0])      # 3-class targets
        probabilities = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8], [0.6, 0.3, 0.1]])
        
        evaluator.update(predictions, targets, probabilities)
        metrics = evaluator.compute_metrics()
        
        # Should have per-class metrics for 3+ classes
        for class_name in multiclass_names:
            assert f'precision_{class_name}' in metrics or f'recall_{class_name}' in metrics or f'f1_{class_name}' in metrics
    
    def test_evaluator_confusion_matrix(self, sample_predictions):
        """Test confusion matrix computation."""
        evaluator = Evaluator()
        
        predictions = np.argmax(sample_predictions['predictions'], axis=1)
        targets = sample_predictions['labels']
        
        evaluator.update(predictions, targets)
        cm = evaluator.get_confusion_matrix()
        
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (evaluator.num_classes, evaluator.num_classes)
        assert cm.sum() == len(predictions)
    
    def test_evaluator_classification_report(self, sample_predictions, class_names):
        """Test classification report generation."""
        evaluator = Evaluator(class_names=class_names)
        
        predictions = np.argmax(sample_predictions['predictions'], axis=1)
        targets = sample_predictions['labels']
        
        evaluator.update(predictions, targets)
        report = evaluator.get_classification_report()
        
        assert isinstance(report, str)
        for class_name in class_names:
            assert class_name in report
    
    def test_evaluator_full_evaluation(self, sample_predictions, class_names):
        """Test complete evaluation workflow."""
        evaluator = Evaluator(class_names=class_names)
        
        predictions = np.argmax(sample_predictions['predictions'], axis=1)
        targets = sample_predictions['labels']
        probabilities = sample_predictions['probabilities']
        
        results = evaluator.evaluate(predictions, targets, probabilities)
        
        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'confusion_matrix' in results
        assert 'classification_report' in results
        assert 'num_samples' in results
        assert 'class_distribution' in results
        
        assert results['num_samples'] == len(predictions)
    
    def test_evaluator_reset(self, sample_predictions):
        """Test evaluator reset functionality."""
        evaluator = Evaluator()
        
        predictions = np.argmax(sample_predictions['predictions'], axis=1)
        targets = sample_predictions['labels']
        
        evaluator.update(predictions, targets)
        assert len(evaluator.all_predictions) > 0
        
        evaluator.reset()
        assert len(evaluator.all_predictions) == 0
        assert len(evaluator.all_targets) == 0
        assert len(evaluator.all_probabilities) == 0
    
    def test_evaluator_save_results(self, sample_predictions, temp_dir):
        """Test saving evaluation results."""
        evaluator = Evaluator()
        
        predictions = np.argmax(sample_predictions['predictions'], axis=1)
        targets = sample_predictions['labels']
        probabilities = sample_predictions['probabilities']
        
        results = evaluator.evaluate(predictions, targets, probabilities)
        
        save_path = temp_dir / "evaluation_results.json"
        evaluator.save_results(results, str(save_path))
        
        assert save_path.exists()
        
        # Check that file contains valid JSON
        import json
        with open(save_path, 'r') as f:
            loaded_results = json.load(f)
            assert 'metrics' in loaded_results
            assert 'num_samples' in loaded_results
    
    def test_evaluator_no_data_error(self):
        """Test error handling when no data is provided."""
        evaluator = Evaluator()
        
        with pytest.raises(ValueError):
            evaluator.compute_metrics()
        
        with pytest.raises(ValueError):
            evaluator.get_confusion_matrix()
        
        with pytest.raises(ValueError):
            evaluator.get_classification_report()


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    @pytest.fixture
    def mock_model(self, mock_hf_model_config):
        """Fixture providing a mock model."""
        from mri_node_validator.models.hf_detector import HFNodeDetector
        model = HFNodeDetector(mock_hf_model_config)
        model.eval()
        return model
    
    def test_model_evaluator_initialization(self, mock_model, device, class_names):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(
            model=mock_model,
            device=device,
            num_classes=2,
            class_names=class_names
        )
        
        assert evaluator.model == mock_model
        assert evaluator.device == device
        assert evaluator.evaluator.num_classes == 2
        assert evaluator.evaluator.class_names == class_names
    
    def test_model_evaluator_single_batch(self, mock_model, sample_batch_dict, device):
        """Test ModelEvaluator with single batch."""
        evaluator = ModelEvaluator(model=mock_model, device=device)
        
        # Mock dataset and dataloader
        from torch.utils.data import DataLoader, Dataset
        
        class MockDataset(Dataset):
            def __init__(self, batch_dict):
                self.batch_dict = batch_dict
                # Extract batch size from labels
                self.batch_size = len(batch_dict['labels'])
            
            def __len__(self):
                return self.batch_size
            
            def __getitem__(self, idx):
                # Return individual sample instead of entire batch
                sample = {}
                for key, value in self.batch_dict.items():
                    if hasattr(value, '__getitem__') and len(value) == self.batch_size:
                        sample[key] = value[idx]
                    else:
                        sample[key] = value
                return sample
        
        dataset = MockDataset(sample_batch_dict)
        dataloader = DataLoader(dataset, batch_size=1)
        
        results = evaluator.evaluate_dataset(dataloader)
        
        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'confusion_matrix' in results
        assert 'num_samples' in results
    
    def test_model_evaluator_multiple_batches(self, mock_model, sample_dataset_items, device):
        """Test ModelEvaluator with multiple batches."""
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
        
        evaluator = ModelEvaluator(model=mock_model, device=device)
        results = evaluator.evaluate_dataset(dataloader)
        
        assert isinstance(results, dict)
        assert results['num_samples'] == len(sample_dataset_items)
    
    def test_model_evaluator_return_predictions(self, mock_model, sample_dataset_items, device):
        """Test ModelEvaluator with return_predictions=True."""
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
        
        evaluator = ModelEvaluator(model=mock_model, device=device)
        results = evaluator.evaluate_dataset(dataloader, return_predictions=True)
        
        assert 'predictions' in results
        assert 'probabilities' in results
        assert isinstance(results['predictions'], np.ndarray)
        assert isinstance(results['probabilities'], np.ndarray)
    
    def test_model_evaluator_device_handling(self, mock_model, sample_dataset_items):
        """Test ModelEvaluator device handling."""
        from torch.utils.data import DataLoader, Dataset
        from mri_node_validator.data.collator import MRIDataCollator
        
        # Test with CPU
        device = torch.device('cpu')
        
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
        
        evaluator = ModelEvaluator(model=mock_model, device=device)
        
        # Should work without device-related errors
        results = evaluator.evaluate_dataset(dataloader)
        assert isinstance(results, dict)
    
    def test_model_evaluator_empty_dataloader(self, mock_model, device):
        """Test ModelEvaluator with empty dataloader."""
        from torch.utils.data import DataLoader, Dataset
        
        class EmptyDataset(Dataset):
            def __len__(self):
                return 0
            
            def __getitem__(self, idx):
                raise IndexError("Empty dataset")
        
        dataset = EmptyDataset()
        dataloader = DataLoader(dataset, batch_size=1)
        
        evaluator = ModelEvaluator(model=mock_model, device=device)
        
        # Should handle empty dataloader gracefully
        try:
            results = evaluator.evaluate_dataset(dataloader)
            # If it succeeds, should have zero samples
            assert results['num_samples'] == 0
        except ValueError:
            # Expected to fail with empty data
            pass


class TestEvaluatorIntegration:
    """Integration tests for evaluation components."""
    
    def test_end_to_end_evaluation_workflow(self, mock_model, sample_dataset_items, temp_dir, device):
        """Test complete evaluation workflow."""
        from torch.utils.data import DataLoader, Dataset
        from mri_node_validator.data.collator import MRIDataCollator
        
        # Setup data
        class MockDataset(Dataset):
            def __init__(self, items):
                self.items = items
            
            def __len__(self):
                return len(self.items)
            
            def __getitem__(self, idx):
                return self.items[idx]
        
        dataset = MockDataset(sample_dataset_items)
        collator = MRIDataCollator()
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)
        
        # Model evaluation
        model_evaluator = ModelEvaluator(model=mock_model, device=device)
        results = model_evaluator.evaluate_dataset(dataloader, return_predictions=True)
        
        # Manual evaluation
        manual_evaluator = Evaluator()
        manual_results = manual_evaluator.evaluate(
            results['predictions'],
            np.array([item['labels'].item() for item in sample_dataset_items]),
            results['probabilities']
        )
        
        # Results should be consistent
        assert results['num_samples'] == manual_results['num_samples']
        
        # Save results
        save_path = temp_dir / "integration_results.json"
        manual_evaluator.save_results(manual_results, str(save_path))
        assert save_path.exists()
    
    def test_evaluator_with_different_metrics(self, sample_predictions):
        """Test evaluator with different averaging strategies."""
        evaluators = [
            Evaluator(average='weighted'),
            Evaluator(average='macro'),
            Evaluator(average='micro')
        ]
        
        predictions = np.argmax(sample_predictions['predictions'], axis=1)
        targets = sample_predictions['labels']
        
        for evaluator in evaluators:
            results = evaluator.evaluate(predictions, targets)
            assert isinstance(results, dict)
            assert 'metrics' in results
    
    def test_evaluator_consistency_check(self, sample_predictions):
        """Test that evaluator results are consistent with sklearn."""
        evaluator = Evaluator()
        
        predictions = np.argmax(sample_predictions['predictions'], axis=1)
        targets = sample_predictions['labels']
        
        results = evaluator.evaluate(predictions, targets)
        
        # Compare with sklearn
        sklearn_accuracy = accuracy_score(targets, predictions)
        sklearn_precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        sklearn_recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        sklearn_f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        assert abs(results['metrics']['accuracy'] - sklearn_accuracy) < 1e-6
        assert abs(results['metrics']['precision'] - sklearn_precision) < 1e-6
        assert abs(results['metrics']['recall'] - sklearn_recall) < 1e-6
        assert abs(results['metrics']['f1'] - sklearn_f1) < 1e-6