#!/usr/bin/env python3
"""
Detailed Inference Example: Load trained model and run comprehensive inference

This script demonstrates:
1. Loading a trained model from local directory or Hugging Face Hub
2. Processing real NIfTI files 
3. Batch inference with evaluation metrics
4. Visualization of results
"""

import torch
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

from mri_node_validator.models.hf_detector import HFNodeDetector
from mri_node_validator.data.dataset import MRIDataset
from mri_node_validator.data.collator import MRIDataCollator
from mri_node_validator.evaluation.evaluator import ModelEvaluator, Evaluator
from mri_node_validator.visualization.plotter import Plotter

def load_nifti_file(file_path: Path) -> np.ndarray:
    """Load a NIfTI file and return the volume data."""
    nifti_img = nib.load(str(file_path))
    return nifti_img.get_fdata()

def preprocess_volume(volume: np.ndarray, target_shape: Tuple[int, int, int] = (64, 64, 64)) -> torch.Tensor:
    """Basic preprocessing of MRI volume."""
    import torch.nn.functional as F
    
    # Convert to tensor and add channel dimension
    volume_tensor = torch.from_numpy(volume).float()
    
    # Normalize intensity
    volume_tensor = (volume_tensor - volume_tensor.mean()) / (volume_tensor.std() + 1e-8)
    
    # Resize to target shape
    volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    volume_tensor = F.interpolate(volume_tensor, size=target_shape, mode='trilinear', align_corners=False)
    volume_tensor = volume_tensor.squeeze(0)  # Remove batch dim, keep channel dim
    
    return volume_tensor

def create_sample_test_data(data_dir: Path, num_samples: int = 10):
    """Create sample test data for demonstration."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test CSV
    file_keys = [f"test_patient_{i:03d}_t1" for i in range(num_samples)]
    labels = np.random.randint(0, 2, num_samples)
    
    df = pd.DataFrame({
        'file_key': file_keys,
        'label': labels,
        'patient_id': [f"TEST_{i:03d}" for i in range(num_samples)],
        'acquisition_date': ['2024-01-01'] * num_samples
    })
    
    csv_path = data_dir / "test_labels.csv"
    df.to_csv(csv_path, index=False)
    
    # Create test NIfTI files
    for i, (file_key, label) in enumerate(zip(file_keys, labels)):
        # Create realistic test volume
        volume = np.random.randn(64, 64, 64).astype(np.float32)
        
        # Add patterns based on label
        if label == 1:  # Merged nodes
            volume[25:40, 25:40, 25:40] += 3.0
            volume[20:45, 20:45, 20:45] += 1.0  # Larger connected region
        else:  # Separate nodes
            volume[20:30, 20:30, 20:30] += 2.0
            volume[35:45, 35:45, 35:45] += 2.0  # Two separate regions
        
        # Save as NIfTI
        nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
        nifti_path = data_dir / f"{file_key}.nii.gz"
        nib.save(nifti_img, str(nifti_path))
    
    return csv_path, df

def run_single_inference(model: HFNodeDetector, 
                        image_path: Path, 
                        segmentation_path: Optional[Path] = None) -> Dict:
    """Run inference on a single MRI volume."""
    
    print(f"🔍 Processing: {image_path.name}")
    
    # Load and preprocess image
    image_volume = load_nifti_file(image_path)
    image_tensor = preprocess_volume(image_volume)
    
    # Load segmentation or create dummy
    if segmentation_path and segmentation_path.exists():
        seg_volume = load_nifti_file(segmentation_path)
        seg_tensor = preprocess_volume(seg_volume)
    else:
        # Create dummy segmentation (in real usage, this should be provided)
        seg_tensor = torch.zeros_like(image_tensor)
        seg_tensor[image_tensor > image_tensor.mean()] = 1.0
    
    # Add batch dimension
    image_batch = image_tensor.unsqueeze(0)
    seg_batch = seg_tensor.unsqueeze(0)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(image=image_batch, segmentation=seg_batch)
        logits = outputs['logits']
        
        # For binary classification
        probabilities = torch.sigmoid(logits)
        predicted_class = (probabilities > 0.5).long()
        confidence = probabilities.item() if predicted_class.item() == 1 else (1 - probabilities.item())
    
    return {
        'file_path': str(image_path),
        'predicted_class': predicted_class.item(),
        'probabilities': probabilities.item(),
        'confidence': confidence,
        'logits': logits.item(),
        'image_shape': image_volume.shape,
        'processed_shape': image_tensor.shape
    }

def run_batch_inference(model: HFNodeDetector, 
                       csv_path: Path, 
                       data_dir: Path,
                       batch_size: int = 4) -> Tuple[List[Dict], Dict]:
    """Run inference on a batch of samples with evaluation."""
    
    print(f"📊 Running batch inference...")
    
    # Setup dataset
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
    
    # Setup dataloader
    collator = MRIDataCollator()
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    # Setup evaluator
    evaluator = ModelEvaluator(
        model=model,
        device='cpu',  # Use CPU for demo
        num_classes=2,
        class_names=["No Merged", "Merged"]
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(dataloader, return_predictions=True)
    
    # Extract individual predictions
    predictions_list = []
    df = pd.read_csv(csv_path)
    
    for i, (pred, prob, true_label) in enumerate(zip(
        results['predictions'], 
        results['probabilities'][:, 1],  # Positive class probability
        [dataset[j]['labels'].item() for j in range(len(dataset))]
    )):
        predictions_list.append({
            'file_key': df.iloc[i]['file_key'],
            'true_label': true_label,
            'predicted_class': pred,
            'probability': prob,
            'confidence': prob if pred == 1 else (1 - prob),
            'correct': pred == true_label
        })
    
    return predictions_list, results

def visualize_results(predictions: List[Dict], 
                     evaluation_results: Dict,
                     output_dir: Path):
    """Create visualizations of inference results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plotter = Plotter(output_dir=output_dir)
    
    print(f"📈 Creating visualizations...")
    
    # Extract data for plotting
    y_true = [p['true_label'] for p in predictions]
    y_pred = [p['predicted_class'] for p in predictions]
    y_scores = [[1-p['probability'], p['probability']] for p in predictions]
    class_names = ["No Merged", "Merged"]
    
    # 1. Confusion Matrix
    cm_fig = plotter.plot_confusion_matrix(
        y_true, y_pred, class_names,
        title="MRI Node Classification Results",
        save_path=str(output_dir / "confusion_matrix.png")
    )
    plt.close(cm_fig)
    
    # 2. ROC Curve
    roc_fig = plotter.plot_roc_curve(
        y_true, y_scores, class_names,
        title="ROC Curve - MRI Node Classification",
        save_path=str(output_dir / "roc_curve.png")
    )
    plt.close(roc_fig)
    
    # 3. Precision-Recall Curve
    pr_fig = plotter.plot_precision_recall_curve(
        y_true, y_scores, class_names,
        title="Precision-Recall Curve",
        save_path=str(output_dir / "precision_recall.png")
    )
    plt.close(pr_fig)
    
    # 4. Class Distribution
    class_counts = {"No Merged": y_true.count(0), "Merged": y_true.count(1)}
    dist_fig = plotter.plot_class_distribution(
        class_counts,
        title="Test Set Class Distribution",
        save_path=str(output_dir / "class_distribution.png")
    )
    plt.close(dist_fig)
    
    print(f"✓ Visualizations saved to {output_dir}")

def main():
    print("🏥 MRI Node Classification - Detailed Inference Example")
    print("=" * 60)
    
    # Setup paths
    model_dir = Path("./trained_model")
    test_data_dir = Path("./test_data")
    output_dir = Path("./inference_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create or use existing test data
    print("\n📁 Setting up test data...")
    if not test_data_dir.exists():
        csv_path, test_df = create_sample_test_data(test_data_dir, num_samples=10)
        print(f"✓ Created sample test data: {len(test_df)} samples")
    else:
        csv_path = test_data_dir / "test_labels.csv"
        test_df = pd.read_csv(csv_path)
        print(f"✓ Using existing test data: {len(test_df)} samples")
    
    # Step 2: Load trained model
    print(f"\n🧠 Loading model from {model_dir}...")
    try:
        model = HFNodeDetector.from_pretrained(str(model_dir))
        print("✓ Model loaded successfully!")
        print(f"  Model config: {model.config.num_labels} classes")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Run the training example first to create a trained model")
        return
    
    # Step 3: Single file inference example
    print(f"\n🔍 Example 1: Single file inference")
    first_file = test_data_dir / f"{test_df.iloc[0]['file_key']}.nii.gz"
    if first_file.exists():
        single_result = run_single_inference(model, first_file)
        
        print(f"  File: {single_result['file_path']}")
        print(f"  Predicted class: {single_result['predicted_class']}")
        print(f"  Confidence: {single_result['confidence']:.1%}")
        print(f"  Raw probability: {single_result['probabilities']:.3f}")
        print(f"  Image shape: {single_result['image_shape']}")
    
    # Step 4: Batch inference with evaluation
    print(f"\n📊 Example 2: Batch inference and evaluation")
    predictions, eval_results = run_batch_inference(model, csv_path, test_data_dir, batch_size=4)
    
    print(f"✓ Processed {len(predictions)} samples")
    print(f"\n📈 Evaluation Results:")
    for metric, value in eval_results['metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    # Step 5: Show detailed predictions
    print(f"\n📋 Sample Predictions:")
    for i, pred in enumerate(predictions[:5]):  # Show first 5
        status = "✓" if pred['correct'] else "❌"
        print(f"  {status} {pred['file_key']}: True={pred['true_label']}, "
              f"Pred={pred['predicted_class']} (conf: {pred['confidence']:.1%})")
    
    # Step 6: Create visualizations
    print(f"\n📊 Creating visualizations...")
    visualize_results(predictions, eval_results, output_dir)
    
    # Step 7: Save detailed results
    results_df = pd.DataFrame(predictions)
    results_csv = output_dir / "detailed_predictions.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"✓ Detailed results saved to {results_csv}")
    
    # Summary
    accuracy = sum(p['correct'] for p in predictions) / len(predictions)
    print(f"\n🎯 Summary:")
    print(f"  Total samples: {len(predictions)}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Results saved to: {output_dir}")
    
    print(f"\n🎉 Inference completed successfully!")

if __name__ == "__main__":
    main()