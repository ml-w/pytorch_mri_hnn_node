#!/usr/bin/env python3
"""
Advanced Training with Custom Augmentation

This script demonstrates:
1. Creating custom augmentation pipelines for 3D MRI data
2. Training with advanced data augmentation techniques
3. Comparing models trained with and without augmentation
4. Using TorchIO for medical image augmentation
"""

import torch
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from transformers import TrainingArguments

# TorchIO imports for medical image augmentation
import torchio as tio

from mri_node_validator.models.hf_detector import HFNodeDetector, MRIConfig
from mri_node_validator.scripts.train import MRITrainer
from mri_node_validator.data.dataset import MRIDataset
from mri_node_validator.data.collator import MRIDataCollator
from mri_node_validator.scripts.train import compute_metrics
from mri_node_validator.evaluation.evaluator import ModelEvaluator
from mri_node_validator.visualization.plotter import Plotter

class CustomMRIDataset(MRIDataset):
    """Extended MRIDataset with custom augmentation support."""
    
    def __init__(self, 
                 csv_path: str,
                 data_dir: str,
                 preprocessing_config: Dict,
                 augmentation_config: Optional[Dict] = None,
                 is_training: bool = True):
        
        super().__init__(csv_path, data_dir, preprocessing_config)
        self.is_training = is_training
        self.augmentation_config = augmentation_config or {}
        
        # Setup augmentation pipeline
        self.setup_augmentation()
    
    def setup_augmentation(self):
        """Setup TorchIO augmentation pipeline."""
        
        if not self.is_training or not self.augmentation_config:
            self.augmentation_transform = None
            return
        
        transforms = []
        
        # Spatial augmentations
        if self.augmentation_config.get('random_flip', False):
            transforms.append(
                tio.RandomFlip(
                    axes=['LR'],  # Left-Right flip
                    flip_probability=0.5
                )
            )
        
        if self.augmentation_config.get('random_rotation', False):
            transforms.append(
                tio.RandomAffine(
                    scales=(0.9, 1.1),
                    degrees=10,
                    translation=2,
                    p=0.5
                )
            )
        
        # Intensity augmentations
        if self.augmentation_config.get('random_gamma', False):
            transforms.append(
                tio.RandomGamma(
                    log_gamma=(-0.3, 0.3),
                    p=0.5
                )
            )
        
        if self.augmentation_config.get('random_noise', False):
            transforms.append(
                tio.RandomNoise(
                    mean=0,
                    std=(0, 0.1),
                    p=0.3
                )
            )
        
        if self.augmentation_config.get('random_blur', False):
            transforms.append(
                tio.RandomBlur(
                    std=(0, 2),
                    p=0.3
                )
            )
        
        # Elastic deformation (advanced)
        if self.augmentation_config.get('elastic_deformation', False):
            transforms.append(
                tio.RandomElasticDeformation(
                    num_control_points=(7, 7, 7),
                    max_displacement=(7.5, 7.5, 7.5),
                    locked_borders=2,
                    p=0.3
                )
            )
        
        # Combine transforms
        if transforms:
            self.augmentation_transform = tio.Compose(transforms)
            print(f"✓ Setup augmentation pipeline with {len(transforms)} transforms")
        else:
            self.augmentation_transform = None
    
    def __getitem__(self, idx):
        """Get item with optional augmentation."""
        
        # Get base item
        item = super().__getitem__(idx)
        
        # Apply augmentation if in training mode
        if self.augmentation_transform and self.is_training:
            try:
                # Create TorchIO subject
                subject = tio.Subject(
                    image=tio.ScalarImage(tensor=item['image'].unsqueeze(0)),  # Add batch dim
                    segmentation=tio.LabelMap(tensor=item['segmentation'].unsqueeze(0))
                )
                
                # Apply augmentation
                augmented_subject = self.augmentation_transform(subject)
                
                # Extract augmented data
                item['image'] = augmented_subject['image'].data.squeeze(0)  # Remove batch dim
                item['segmentation'] = augmented_subject['segmentation'].data.squeeze(0)
                
            except Exception as e:
                print(f"Warning: Augmentation failed for sample {idx}: {e}")
                # Return original item if augmentation fails
                pass
        
        return item

def create_enhanced_training_data(data_dir: Path, num_samples: int = 50):
    """Create enhanced training data with more variety."""
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create more realistic labels with additional metadata
    file_keys = [f"patient_{i:03d}_t1" for i in range(num_samples)]
    
    # Create imbalanced dataset (more realistic)
    labels = np.random.choice([0, 1], num_samples, p=[0.7, 0.3])  # 70% no-merged, 30% merged
    
    # Additional metadata
    ages = np.random.normal(50, 15, num_samples).astype(int)
    ages = np.clip(ages, 18, 90)
    sexes = np.random.choice(['M', 'F'], num_samples)
    scanner_types = np.random.choice(['Siemens', 'GE', 'Philips'], num_samples)
    field_strengths = np.random.choice(['1.5T', '3T'], num_samples, p=[0.3, 0.7])
    
    df = pd.DataFrame({
        'file_key': file_keys,
        'label': labels,
        'age': ages,
        'sex': sexes,
        'scanner_type': scanner_types,
        'field_strength': field_strengths,
        'acquisition_date': pd.date_range('2020-01-01', periods=num_samples, freq='D')
    })
    
    csv_path = data_dir / "enhanced_labels.csv"
    df.to_csv(csv_path, index=False)
    
    # Create more realistic NIfTI files
    for i, (file_key, label, age) in enumerate(zip(file_keys, labels, ages)):
        # Base volume with age-related changes
        volume = np.random.randn(64, 64, 64).astype(np.float32)
        
        # Add age-related atrophy (older patients have more space)
        age_factor = (age - 20) / 70  # Normalize age to 0-1
        volume *= (1 - age_factor * 0.2)  # Slight volume reduction with age
        
        # Add anatomical structure
        # Brain-like structure
        xx, yy, zz = np.meshgrid(
            np.linspace(-1, 1, 64),
            np.linspace(-1, 1, 64), 
            np.linspace(-1, 1, 64)
        )
        brain_mask = (xx**2 + yy**2 + zz**2) < 0.8
        volume[brain_mask] += 1.0
        
        # Add lymph node regions
        if label == 1:  # Merged nodes
            # Larger, more connected regions
            center_x, center_y, center_z = 32, 32, 32
            for dx, dy, dz in [(0, 0, 0), (5, 5, 5), (-5, -5, -5)]:
                x, y, z = center_x + dx, center_y + dy, center_z + dz
                volume[x-8:x+8, y-8:y+8, z-8:z+8] += 2.0
        else:  # Separate nodes
            # Smaller, separate regions
            centers = [(25, 25, 25), (40, 40, 40), (30, 45, 30)]
            for cx, cy, cz in centers:
                volume[cx-4:cx+4, cy-4:cy+4, cz-4:cz+4] += 1.5
        
        # Add noise based on scanner type
        noise_level = {'Siemens': 0.1, 'GE': 0.12, 'Philips': 0.08}
        scanner = scanner_types[i]
        volume += np.random.normal(0, noise_level[scanner], volume.shape)
        
        # Save as NIfTI
        nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
        nifti_path = data_dir / f"{file_key}.nii.gz"
        nib.save(nifti_img, str(nifti_path))
    
    print(f"✓ Created enhanced dataset: {len(df)} samples")
    print(f"  Class distribution: {df['label'].value_counts().to_dict()}")
    print(f"  Age range: {df['age'].min()}-{df['age'].max()}")
    print(f"  Scanner types: {df['scanner_type'].value_counts().to_dict()}")
    
    return csv_path, df

def train_model_with_config(model_config: MRIConfig,
                           train_dataset,
                           eval_dataset,
                           output_dir: Path,
                           experiment_name: str,
                           num_epochs: int = 3) -> Dict:
    """Train a model with given configuration."""
    
    print(f"\n🚀 Training model: {experiment_name}")
    
    # Initialize model
    model = HFNodeDetector(model_config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / experiment_name),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=5,
        weight_decay=0.01,
        learning_rate=model_config.learning_rate,
        logging_dir=str(output_dir / experiment_name / "logs"),
        logging_steps=5,
        eval_steps=10,
        save_steps=20,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=None,
    )
    
    # Data collator
    data_collator = MRIDataCollator()
    
    # Trainer
    trainer = MRITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    train_result = trainer.train()
    
    # Final evaluation
    eval_result = trainer.evaluate()
    
    # Save model
    trainer.save_model()
    
    return {
        'model': model,
        'trainer': trainer,
        'train_result': train_result,
        'eval_result': eval_result,
        'model_path': output_dir / experiment_name
    }

def compare_models(results: Dict[str, Dict], test_dataset, output_dir: Path):
    """Compare different trained models."""
    
    print(f"\n📊 Comparing models...")
    
    comparison_results = {}
    
    for experiment_name, result in results.items():
        print(f"\nEvaluating {experiment_name}...")
        
        # Load best model
        model = HFNodeDetector.from_pretrained(str(result['model_path']))
        
        # Setup evaluator
        evaluator = ModelEvaluator(
            model=model,
            device='cpu',
            num_classes=2,
            class_names=["No Merged", "Merged"]
        )
        
        # Create test dataloader
        data_collator = MRIDataCollator()
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0
        )
        
        # Evaluate
        test_results = evaluator.evaluate_dataset(test_dataloader, return_predictions=True)
        
        comparison_results[experiment_name] = {
            'test_metrics': test_results['metrics'],
            'eval_metrics': result['eval_result'],
            'predictions': test_results['predictions'],
            'probabilities': test_results['probabilities']
        }
        
        print(f"  Test Accuracy: {test_results['metrics']['accuracy']:.4f}")
        print(f"  Test F1: {test_results['metrics']['f1']:.4f}")
    
    # Create comparison visualization
    create_comparison_plots(comparison_results, output_dir)
    
    return comparison_results

def create_comparison_plots(comparison_results: Dict, output_dir: Path):
    """Create plots comparing different models."""
    
    plots_dir = output_dir / "comparison_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plotter = Plotter(output_dir=plots_dir)
    
    # 1. Metrics comparison
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    model_names = list(comparison_results.keys())
    metrics_data = {metric: [] for metric in metrics_to_compare}
    
    for model_name in model_names:
        test_metrics = comparison_results[model_name]['test_metrics']
        for metric in metrics_to_compare:
            metrics_data[metric].append(test_metrics.get(metric, 0))
    
    # Create metrics comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, metric in enumerate(metrics_to_compare):
        ax.bar(x + i * width, metrics_data[metric], width, label=metric.upper())
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC curves comparison
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, results in comparison_results.items():
        # Extract test set true labels (assuming same test set for all models)
        if 'test_dataset' in globals():
            y_true = [test_dataset[i]['labels'].item() for i in range(len(test_dataset))]
        else:
            # Fallback - reconstruct from first model
            y_true = [0, 1] * (len(results['predictions']) // 2)  # Dummy
        
        y_scores = results['probabilities'][:, 1]  # Positive class probabilities
        
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plots saved to {plots_dir}")

def main():
    print("🔬 Advanced MRI Training with Custom Augmentation")
    print("=" * 60)
    
    # Setup
    data_dir = Path("./augmentation_data")
    output_dir = Path("./augmentation_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create enhanced training data
    print("\n📁 Creating enhanced training dataset...")
    csv_path, train_df = create_enhanced_training_data(data_dir, num_samples=60)
    
    # Step 2: Setup different augmentation configurations
    augmentation_configs = {
        'no_augmentation': {},
        
        'basic_augmentation': {
            'random_flip': True,
            'random_noise': True,
        },
        
        'advanced_augmentation': {
            'random_flip': True,
            'random_rotation': True,
            'random_gamma': True,
            'random_noise': True,
            'random_blur': True,
        },
        
        'full_augmentation': {
            'random_flip': True,
            'random_rotation': True,
            'random_gamma': True,
            'random_noise': True,
            'random_blur': True,
            'elastic_deformation': True,
        }
    }
    
    # Step 3: Create datasets for each configuration
    preprocessing_config = {
        'target_spacing': [1.0, 1.0, 1.0],
        'target_shape': [64, 64, 64],
        'intensity_clipping': True,
        'clipping_percentiles': [1.0, 99.0]
    }
    
    # Split data
    total_samples = len(train_df)
    train_size = int(0.7 * total_samples)
    val_size = int(0.2 * total_samples)
    test_size = total_samples - train_size - val_size
    
    print(f"\nDataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Step 4: Train models with different augmentation strategies
    print(f"\n🎯 Training models with different augmentation strategies...")
    
    experiment_results = {}
    
    for aug_name, aug_config in augmentation_configs.items():
        print(f"\n{'='*20} {aug_name.upper()} {'='*20}")
        
        # Create datasets
        full_dataset = CustomMRIDataset(
            csv_path=str(csv_path),
            data_dir=str(data_dir),
            preprocessing_config=preprocessing_config,
            augmentation_config=aug_config,
            is_training=True
        )
        
        # Split dataset
        train_dataset, temp_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size + test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        val_dataset, test_dataset = torch.utils.data.random_split(
            temp_dataset, [val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Model configuration
        model_config = MRIConfig(
            learning_rate=1e-3 if aug_name == 'no_augmentation' else 5e-4,  # Lower LR for augmented
            architecture_type="3d_resnet_attention",
            num_labels=2,
        )
        
        # Train model
        result = train_model_with_config(
            model_config=model_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            output_dir=output_dir,
            experiment_name=aug_name,
            num_epochs=3
        )
        
        experiment_results[aug_name] = result
        
        print(f"✓ {aug_name} completed!")
        print(f"  Final eval accuracy: {result['eval_result']['eval_accuracy']:.4f}")
        print(f"  Final eval F1: {result['eval_result']['eval_f1']:.4f}")
    
    # Step 5: Compare all models on test set
    print(f"\n🔍 Comparing models on test set...")
    
    # Use test dataset from last split (no augmentation for fair comparison)
    test_dataset_clean = CustomMRIDataset(
        csv_path=str(csv_path),
        data_dir=str(data_dir),
        preprocessing_config=preprocessing_config,
        augmentation_config={},  # No augmentation for testing
        is_training=False
    )
    
    # Extract test indices (same split as before)
    _, temp_dataset = torch.utils.data.random_split(
        test_dataset_clean, [train_size, val_size + test_size],
        generator=torch.Generator().manual_seed(42)
    )
    _, test_dataset_final = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    comparison_results = compare_models(experiment_results, test_dataset_final, output_dir)
    
    # Step 6: Summary report
    print(f"\n📊 EXPERIMENT SUMMARY")
    print("=" * 50)
    
    best_model = None
    best_f1 = 0
    
    for exp_name, comp_result in comparison_results.items():
        test_metrics = comp_result['test_metrics']
        print(f"\n{exp_name.upper()}:")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test Precision: {test_metrics['precision']:.4f}")
        print(f"  Test Recall: {test_metrics['recall']:.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f}")
        print(f"  Test AUC: {test_metrics.get('auc', 0):.4f}")
        
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            best_model = exp_name
    
    print(f"\n🏆 Best performing model: {best_model.upper()}")
    print(f"   F1 Score: {best_f1:.4f}")
    
    print(f"\n📁 All results saved to: {output_dir}")
    print(f"🎉 Augmentation experiment completed!")

if __name__ == "__main__":
    main()