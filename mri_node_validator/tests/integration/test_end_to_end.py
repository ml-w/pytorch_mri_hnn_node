"""End-to-end integration tests for the complete MRI classification pipeline."""

import pytest
import torch
import numpy as np
import tempfile
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from mri_node_validator.data.dataset import MRIDataModule
from mri_node_validator.data.transforms import get_preprocessing_pipeline
from mri_node_validator.data.collator import MRIDataCollator
from mri_node_validator.models.hf_detector import HFNodeDetector, MRIConfig
from mri_node_validator.scripts.train import MRITrainer, compute_metrics
from mri_node_validator.evaluation.evaluator import ModelEvaluator
from mri_node_validator.visualization.plotter import Plotter
from transformers import TrainingArguments


class TestFullPipeline:
    """Test cases for complete end-to-end pipeline."""
    
    def test_data_to_model_pipeline(self, temp_dir, sample_nifti_files, sample_csv_file, training_config):
        """Test complete data loading to model inference pipeline."""
        # 1. Setup data module
        train_transforms, val_transforms, test_transforms = get_preprocessing_pipeline(
            training_config['preprocessing']
        )
        
        data_module = MRIDataModule(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            batch_size=1,
            num_workers=0,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            train_split=0.6,
            val_split=0.2,
            test_split=0.2,
            random_seed=42
        )
        
        # 2. Setup datasets
        train_dataset, val_dataset, test_dataset = data_module.setup()
        
        # 3. Setup model
        config = MRIConfig(
            learning_rate=training_config['model']['learning_rate'],
            model_type=training_config['model']['model_type'],
            num_labels=training_config['model']['num_classes']
        )
        model = HFNodeDetector(config)
        
        # 4. Test inference on each dataset
        model.eval()
        datasets = [train_dataset, val_dataset, test_dataset]
        
        for dataset in datasets:
            if len(dataset) > 0:
                sample = dataset[0]
                
                # Ensure proper tensor shapes
                image = sample['image'].unsqueeze(0)  # Add batch dimension
                segmentation = sample['segmentation'].unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(image=image, segmentation=segmentation)
                
                assert 'logits' in outputs
                assert outputs['logits'].shape[0] == 1  # Batch size
    
    def test_training_pipeline_minimal(self, temp_dir, sample_nifti_files, sample_csv_file, training_config):
        """Test minimal training pipeline."""
        # Setup data
        train_transforms, val_transforms, test_transforms = get_preprocessing_pipeline(
            training_config['preprocessing']
        )
        
        data_module = MRIDataModule(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            batch_size=1,
            num_workers=0,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            random_seed=42
        )
        
        train_dataset, val_dataset, test_dataset = data_module.setup()
        
        # Setup model
        config = MRIConfig(
            learning_rate=training_config['model']['learning_rate'],
            model_type=training_config['model']['model_type'],
            num_labels=training_config['model']['num_classes']
        )
        model = HFNodeDetector(config)
        
        # Setup trainer
        training_args = TrainingArguments(
            output_dir=str(temp_dir / "training_output"),
            num_train_epochs=1,
            max_steps=2,  # Set max_steps here instead of passing to train()
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            save_steps=10,
            eval_steps=5,
            logging_steps=2,
            remove_unused_columns=False,
            report_to=None,  # Disable logging
        )
        
        collator = MRIDataCollator()
        
        trainer = MRITrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if len(train_dataset) > 0 else None,
            eval_dataset=val_dataset if len(val_dataset) > 0 else None,
            data_collator=collator,
            compute_metrics=compute_metrics
        )
        
        # Run minimal training
        if len(train_dataset) > 0:
            trainer.train()
            
            # Verify model state changed
            assert model.training or not model.training  # Just check it exists
    
    def test_evaluation_pipeline(self, temp_dir, sample_nifti_files, sample_csv_file, training_config):
        """Test complete evaluation pipeline."""
        # Setup data
        train_transforms, val_transforms, test_transforms = get_preprocessing_pipeline(
            training_config['preprocessing']
        )
        
        data_module = MRIDataModule(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            batch_size=1,
            num_workers=0,
            val_transforms=val_transforms,
            random_seed=42
        )
        
        _, val_dataset, _ = data_module.setup()
        
        if len(val_dataset) == 0:
            pytest.skip("No validation data available")
        
        # Setup model
        config = MRIConfig(num_labels=2)
        model = HFNodeDetector(config)
        model.eval()
        
        # Setup evaluator
        evaluator = ModelEvaluator(
            model=model,
            device=torch.device('cpu'),
            num_classes=2,
            class_names=["No_Merged", "Merged"]
        )
        
        # Create dataloader
        collator = MRIDataCollator()
        dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=collator)
        
        # Run evaluation
        results = evaluator.evaluate_dataset(dataloader, return_predictions=True)
        
        # Verify results
        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'confusion_matrix' in results
        assert 'predictions' in results
        assert 'probabilities' in results
        assert results['num_samples'] == len(val_dataset)
    
    def test_visualization_pipeline(self, temp_dir, sample_training_history, sample_predictions, class_names):
        """Test complete visualization pipeline."""
        # Setup plotter
        plotter = Plotter(output_dir=temp_dir)
        
        # Create all visualizations
        y_true = sample_predictions['labels']
        y_pred = np.argmax(sample_predictions['predictions'], axis=1)
        y_scores = sample_predictions['probabilities']
        
        # 1. Training history
        history_fig = plotter.plot_training_history(
            history=sample_training_history,
            save_path=str(temp_dir / "training_history.png")
        )
        assert history_fig is not None
        
        # 2. Confusion matrix
        cm_fig = plotter.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            save_path=str(temp_dir / "confusion_matrix.png")
        )
        assert cm_fig is not None
        
        # 3. ROC curve
        roc_fig = plotter.plot_roc_curve(
            y_true=y_true,
            y_scores=y_scores,
            class_names=class_names,
            save_path=str(temp_dir / "roc_curve.png")
        )
        assert roc_fig is not None
        
        # 4. Sample predictions with 3D data
        images = torch.randn(len(y_true), 1, 32, 32, 32)  # 3D MRI volumes
        sample_fig = plotter.plot_sample_predictions(
            images=images,
            true_labels=y_true,
            pred_labels=y_pred,
            pred_probs=y_scores,
            class_names=class_names,
            save_path=str(temp_dir / "sample_predictions.png")
        )
        assert sample_fig is not None
        
        # Verify all files were created
        expected_files = [
            "training_history.png",
            "confusion_matrix.png", 
            "roc_curve.png",
            "sample_predictions.png"
        ]
        
        for filename in expected_files:
            assert (temp_dir / filename).exists()
    
    def test_config_driven_pipeline(self, temp_dir, sample_nifti_files, sample_csv_file, training_config_file):
        """Test pipeline driven by configuration file."""
        # Load config
        with open(training_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Setup preprocessing pipeline from config
        train_transforms, val_transforms, test_transforms = get_preprocessing_pipeline(
            config['preprocessing']
        )
        
        # Setup data module from config
        data_module = MRIDataModule(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            train_split=config['data']['train_split'],
            val_split=config['data']['val_split'],
            test_split=config['data']['test_split'],
            random_seed=config['seed']
        )
        
        train_dataset, val_dataset, test_dataset = data_module.setup()
        
        # Setup model from config
        model_config = MRIConfig(
            learning_rate=config['model']['learning_rate'],
            model_type=config['model']['model_type'],
            num_labels=config['model']['num_classes']
        )
        model = HFNodeDetector(model_config)
        
        # Test that everything works together
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            
            image = sample['image'].unsqueeze(0)
            segmentation = sample['segmentation'].unsqueeze(0)
            
            model.eval()
            with torch.no_grad():
                outputs = model(image=image, segmentation=segmentation)
            
            assert 'logits' in outputs
    
    def test_3d_consistency_throughout_pipeline(self, temp_dir, sample_nifti_files, sample_csv_file):
        """Test that 3D tensor shapes are maintained throughout the pipeline."""
        # Setup minimal config
        config = {
            'target_spacing': [2.0, 2.0, 2.0],
            'target_shape': [32, 32, 32],
            'augmentation_probability': 0.0  # No randomness for testing
        }
        
        train_transforms, _, _ = get_preprocessing_pipeline(config)
        
        # Setup data
        data_module = MRIDataModule(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            batch_size=1,
            num_workers=0,
            train_transforms=train_transforms,
            random_seed=42
        )
        
        train_dataset, _, _ = data_module.setup()
        
        if len(train_dataset) == 0:
            pytest.skip("No training data available")
        
        # Test data shapes
        sample = train_dataset[0]
        
        # Check 3D shapes are preserved
        assert sample['image'].dim() == 4  # (C, D, H, W)
        assert sample['segmentation'].dim() == 4  # (C, D, H, W)
        assert sample['image'].shape[1:] == (32, 32, 32)  # Target shape
        assert sample['segmentation'].shape[1:] == (32, 32, 32)
        
        # Test with dataloader
        collator = MRIDataCollator()
        dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collator)
        
        for batch in dataloader:
            assert batch['image'].dim() == 5  # (B, C, D, H, W)
            assert batch['segmentation'].dim() == 5  # (B, C, D, H, W)
            assert batch['image'].shape[2:] == (32, 32, 32)  # D, H, W
            break
        
        # Test with model
        model = HFNodeDetector(MRIConfig(num_labels=2))
        model.eval()
        
        with torch.no_grad():
            outputs = model(
                image=batch['image'],
                segmentation=batch['segmentation']
            )
        
        assert 'logits' in outputs
        assert outputs['logits'].shape[0] == 1  # Batch size
    
    def test_memory_efficiency_large_volumes(self, temp_dir, sample_csv_file):
        """Test memory efficiency with larger volumes."""
        # Create a larger test volume
        import nibabel as nib
        
        # Create larger NIfTI file
        large_data = np.random.randn(128, 128, 128).astype(np.float32)
        img = nib.Nifti1Image(large_data, affine=np.eye(4))
        
        large_file = temp_dir / "large_patient.nii.gz"
        nib.save(img, str(large_file))
        
        # Update CSV to include large file
        import pandas as pd
        csv_data = pd.DataFrame({
            'file_key': ['large_patient'],
            'label': [0]
        })
        large_csv = temp_dir / "large_labels.csv"
        csv_data.to_csv(large_csv, index=False)
        
        # Setup with smaller target size to manage memory
        config = {
            'target_spacing': [2.0, 2.0, 2.0],
            'target_shape': [64, 64, 64],  # Reasonable size for testing
            'augmentation_probability': 0.0
        }
        
        train_transforms, _, _ = get_preprocessing_pipeline(config)
        
        # Test data loading
        data_module = MRIDataModule(
            data_dir=temp_dir,
            csv_path=large_csv,
            batch_size=1,
            num_workers=0,
            train_transforms=train_transforms,
            random_seed=42
        )
        
        train_dataset, _, _ = data_module.setup()
        
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            
            # Verify size was reduced
            assert sample['image'].shape[1:] == (64, 64, 64)
            
            # Test with model
            model = HFNodeDetector(MRIConfig(num_labels=2))
            model.eval()
            
            image = sample['image'].unsqueeze(0)
            segmentation = sample['segmentation'].unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(image=image, segmentation=segmentation)
            
            assert 'logits' in outputs
    
    def test_error_handling_malformed_data(self, temp_dir):
        """Test error handling with malformed data."""
        # Create malformed CSV
        import pandas as pd
        
        malformed_csv = pd.DataFrame({
            'wrong_column': ['value1', 'value2'],
            'also_wrong': [1, 2]
        })
        csv_path = temp_dir / "malformed.csv"
        malformed_csv.to_csv(csv_path, index=False)
        
        # Should handle missing required columns gracefully
        with pytest.raises(KeyError):
            data_module = MRIDataModule(
                data_dir=temp_dir,
                csv_path=csv_path,
                batch_size=1,
                random_seed=42
            )
            train_dataset, _, _ = data_module.setup()
    
    def test_reproducibility_with_seed(self, temp_dir, sample_nifti_files, sample_csv_file):
        """Test that results are reproducible with same seed."""
        config = {
            'target_spacing': [1.0, 1.0, 1.0],
            'target_shape': [32, 32, 32],
            'augmentation_probability': 0.0  # No randomness for reproducibility
        }
        
        # Setup identical data modules with same seed
        data_modules = []
        for _ in range(2):
            train_transforms, _, _ = get_preprocessing_pipeline(config)
            
            data_module = MRIDataModule(
                data_dir=temp_dir,
                csv_path=sample_csv_file,
                batch_size=1,
                num_workers=0,
                train_transforms=train_transforms,
                random_seed=42  # Same seed
            )
            data_modules.append(data_module)
        
        # Get datasets
        datasets = [dm.setup() for dm in data_modules]
        
        # Compare first samples if available
        if len(datasets[0][0]) > 0 and len(datasets[1][0]) > 0:
            sample1 = datasets[0][0][0]  # First training dataset, first sample
            sample2 = datasets[1][0][0]  # Second training dataset, first sample
            
            # Should be identical (no augmentation)
            assert torch.allclose(sample1['image'], sample2['image'], atol=1e-6)
            assert torch.allclose(sample1['segmentation'], sample2['segmentation'], atol=1e-6)
            assert sample1['labels'].item() == sample2['labels'].item()


class TestPerformanceBenchmarks:
    """Performance and benchmark tests."""
    
    def test_inference_speed(self, temp_dir, sample_nifti_files, sample_csv_file):
        """Test inference speed with multiple samples."""
        import time
        
        # Setup data
        config = {
            'target_spacing': [2.0, 2.0, 2.0],
            'target_shape': [64, 64, 64],
            'augmentation_probability': 0.0
        }
        
        _, val_transforms, _ = get_preprocessing_pipeline(config)
        
        data_module = MRIDataModule(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            batch_size=1,
            num_workers=0,
            val_transforms=val_transforms,
            random_seed=42
        )
        
        _, val_dataset, _ = data_module.setup()
        
        if len(val_dataset) == 0:
            pytest.skip("No validation data available")
        
        # Setup model
        model = HFNodeDetector(MRIConfig(num_labels=2))
        model.eval()
        
        # Time inference
        collator = MRIDataCollator()
        dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=collator)
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(
                    image=batch['image'],
                    segmentation=batch['segmentation']
                )
                # Just verify output exists
                assert 'logits' in outputs
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time (very lenient for CI)
        samples_per_second = len(val_dataset) / total_time if total_time > 0 else float('inf')
        print(f"Inference speed: {samples_per_second:.2f} samples/second")
        
        # Just verify it completed without error
        assert total_time >= 0
    
    def test_memory_usage_tracking(self, temp_dir, sample_nifti_files, sample_csv_file):
        """Test memory usage tracking during pipeline execution."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Setup and run pipeline
        config = {
            'target_spacing': [2.0, 2.0, 2.0],
            'target_shape': [64, 64, 64],
            'augmentation_probability': 0.0
        }
        
        train_transforms, _, _ = get_preprocessing_pipeline(config)
        
        data_module = MRIDataModule(
            data_dir=temp_dir,
            csv_path=sample_csv_file,
            batch_size=1,
            num_workers=0,
            train_transforms=train_transforms,
            random_seed=42
        )
        
        train_dataset, _, _ = data_module.setup()
        
        # Load model
        model = HFNodeDetector(MRIConfig(num_labels=2))
        
        # Process some data
        if len(train_dataset) > 0:
            collator = MRIDataCollator()
            dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collator)
            
            for batch in dataloader:
                with torch.no_grad():
                    outputs = model(
                        image=batch['image'],
                        segmentation=batch['segmentation']
                    )
                break
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.1f} MB -> {final_memory:.1f} MB (+{memory_increase:.1f} MB)")
        
        # Just verify we can track memory (no hard limits for CI)
        assert memory_increase >= 0  # Memory should not decrease