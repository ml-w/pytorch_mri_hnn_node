#!/usr/bin/env python3
"""
Basic Training Example: Train a 3D ResNet for MRI node classification

This script demonstrates:
1. Setting up training data with CSV labels
2. Training a 3D ResNet model from scratch
3. Saving the trained model to Hugging Face format
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import TrainingArguments
from torch.utils.data import DataLoader

from mri_node_validator.models.hf_detector import HFNodeDetector, MRIConfig
from mri_node_validator.scripts.train import MRITrainer
from mri_node_validator.data.dataset import MRIDataset
from mri_node_validator.data.collator import MRIDataCollator
from mri_node_validator.scripts.train import compute_metrics

def create_sample_data(data_dir: Path, num_samples: int = 20):
    """Create sample MRI data and labels for demonstration."""
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample CSV labels
    file_keys = [f"patient_{i:03d}_t1" for i in range(num_samples)]
    labels = np.random.randint(0, 2, num_samples)  # Binary labels: 0=no merged, 1=merged
    ages = np.random.randint(20, 80, num_samples)
    sexes = np.random.choice(['M', 'F'], num_samples)
    
    df = pd.DataFrame({
        'file_key': file_keys,
        'label': labels,
        'age': ages,
        'sex': sexes
    })
    
    csv_path = data_dir / "labels.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Created sample labels: {csv_path}")
    
    # Create dummy NIfTI files (in real usage, these would be actual MRI scans)
    import nibabel as nib
    
    nifti_files = []
    for file_key in file_keys:
        # Create 3D volume with some structure
        volume = np.random.randn(64, 64, 64).astype(np.float32)
        
        # Add some realistic MRI-like structure
        if labels[file_keys.index(file_key)] == 1:  # Merged nodes
            # Add larger connected regions for merged cases
            volume[20:40, 20:40, 20:40] += 2.0
        else:  # Separate nodes
            # Add smaller separate regions
            volume[15:25, 15:25, 15:25] += 1.5
            volume[35:45, 35:45, 35:45] += 1.5
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
        nifti_path = data_dir / f"{file_key}.nii.gz"
        nib.save(nifti_img, str(nifti_path))
        nifti_files.append(nifti_path)
    
    print(f"✓ Created {len(nifti_files)} sample NIfTI files")
    return csv_path, nifti_files

def main():
    # Setup
    data_dir = Path("./sample_data")
    output_dir = Path("./trained_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔬 MRI Node Classification Training Example")
    print("=" * 50)
    
    # Step 1: Create sample data
    print("\n📁 Creating sample training data...")
    csv_path, nifti_files = create_sample_data(data_dir, num_samples=20)
    
    # Step 2: Setup model configuration
    print("\n🧠 Setting up model configuration...")
    config = MRIConfig(
        learning_rate=1e-3,
        architecture_type="3d_resnet_attention",
        num_labels=2,  # Binary classification
    )
    
    # Initialize model
    model = HFNodeDetector(config)
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Step 3: Setup dataset
    print("\n📊 Setting up training dataset...")
    dataset = MRIDataset(
        csv_path=str(csv_path),
        data_dir=str(data_dir),
        preprocessing_config={
            'target_spacing': [1.0, 1.0, 1.0],
            'target_shape': [64, 64, 64],
            'intensity_clipping': True,
            'clipping_percentiles': [1.0, 99.0]
        }
    )
    
    print(f"✓ Dataset created with {len(dataset)} samples")
    print(f"  Class distribution: {dataset.get_class_distribution()}")
    
    # Step 4: Setup training arguments
    print("\n⚙️ Configuring training parameters...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=2,  # Short training for demo
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=5,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=2,
        eval_steps=5,
        save_steps=10,
        evaluation_strategy="steps",
        save_strategy="steps",
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        report_to=None,  # Disable wandb/tensorboard for demo
    )
    
    # Step 5: Setup data splitting and training
    print("\n🎯 Setting up trainer...")
    
    # Simple train/eval split
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Data collator
    data_collator = MRIDataCollator()
    
    # Initialize trainer
    trainer = MRITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print(f"✓ Trainer setup complete")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Evaluation samples: {len(eval_dataset)}")
    
    # Step 6: Train the model
    print("\n🚀 Starting training...")
    try:
        train_result = trainer.train()
        print("✓ Training completed successfully!")
        print(f"  Final training loss: {train_result.training_loss:.4f}")
        
        # Step 7: Evaluate the model
        print("\n📈 Evaluating model...")
        eval_result = trainer.evaluate()
        print("✓ Evaluation completed!")
        for key, value in eval_result.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        # Step 8: Save the trained model
        print("\n💾 Saving trained model...")
        trainer.save_model()
        print(f"✓ Model saved to {output_dir}")
        
        # Step 9: Test loading the saved model
        print("\n🔄 Testing model loading...")
        loaded_model = HFNodeDetector.from_pretrained(str(output_dir))
        print("✓ Model loaded successfully!")
        
        # Quick inference test
        print("\n🔍 Testing inference...")
        sample = dataset[0]
        image = sample['image'].unsqueeze(0)  # Add batch dimension
        segmentation = sample['segmentation'].unsqueeze(0)
        
        loaded_model.eval()
        with torch.no_grad():
            outputs = loaded_model(image=image, segmentation=segmentation)
            probabilities = torch.sigmoid(outputs['logits'])
            predicted_class = (probabilities > 0.5).long()
            
            print(f"  Sample prediction: {predicted_class.item()}")
            print(f"  Confidence: {probabilities.item():.3f}")
            print(f"  True label: {sample['labels'].item()}")
        
        print("\n🎉 Training example completed successfully!")
        print(f"\nTo use your trained model:")
        print(f"  model = HFNodeDetector.from_pretrained('{output_dir}')")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()