"""Evaluator module for MRI classification metrics and validation."""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from .metrics import ClassificationMetrics


class Evaluator:
    """Main evaluator class for MRI classification validation."""
    
    def __init__(
        self,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None,
        average: str = 'weighted'
    ):
        """
        Initialize the evaluator.
        
        Args:
            num_classes: Number of classes
            class_names: Names of the classes
            average: Averaging strategy for multi-class metrics
        """
        # If class_names provided but num_classes not specified, infer from class_names
        if class_names is not None and num_classes == 2:
            num_classes = len(class_names)
        
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.average = average
        
        # Initialize metrics collection
        self.metrics = ClassificationMetrics(num_classes=num_classes, class_names=class_names)
        
        # Storage for predictions and targets
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None
    ):
        """
        Update evaluator with new predictions and targets.
        
        Args:
            predictions: Model predictions (class indices)
            targets: Ground truth labels
            probabilities: Class probabilities (optional)
        """
        # Convert to numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        # Store for later computation
        self.all_predictions.extend(predictions.flatten())
        self.all_targets.extend(targets.flatten())
        
        if probabilities is not None:
            if probabilities.ndim == 1:
                # Binary classification with single probability
                probabilities = np.column_stack([1 - probabilities, probabilities])
            self.all_probabilities.extend(probabilities)
        
        # Update torchmetrics with proper format
        if probabilities is not None:
            probs_tensor = torch.tensor(probabilities)
            targets_tensor = torch.tensor(targets)
            
            # Handle probability format based on metrics type
            if self.num_classes == 2:
                # Binary classification - use single probability (positive class)
                if probs_tensor.ndim == 2 and probs_tensor.shape[1] == 2:
                    probs_tensor = probs_tensor[:, 1]  # Use positive class probability
                elif probs_tensor.ndim == 2 and probs_tensor.shape[1] == 1:
                    probs_tensor = probs_tensor.squeeze(1)  # Squeeze single column
            # For multiclass (3+ classes), keep full probability tensor
            
            self.metrics.update(probs_tensor, targets_tensor)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        if not self.all_predictions or not self.all_targets:
            raise ValueError("No predictions or targets available. Call update() first.")
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        metrics['precision'] = precision_score(targets, predictions, average=self.average, zero_division=0)
        metrics['recall'] = recall_score(targets, predictions, average=self.average, zero_division=0)
        metrics['f1'] = f1_score(targets, predictions, average=self.average, zero_division=0)
        
        # Per-class metrics (only for multiclass with 3+ classes)
        if self.num_classes > 2:
            for i, class_name in enumerate(self.class_names):
                metrics[f'precision_{class_name}'] = precision_score(
                    targets, predictions, labels=[i], average=None, zero_division=0
                )[0] if i in targets else 0.0
                metrics[f'recall_{class_name}'] = recall_score(
                    targets, predictions, labels=[i], average=None, zero_division=0
                )[0] if i in targets else 0.0
                metrics[f'f1_{class_name}'] = f1_score(
                    targets, predictions, labels=[i], average=None, zero_division=0
                )[0] if i in targets else 0.0
        
        # AUC metrics (if probabilities available)
        if self.all_probabilities:
            probabilities = np.array(self.all_probabilities)
            
            if self.num_classes == 2:
                # Binary classification
                if probabilities.shape[1] == 2:
                    metrics['auc'] = roc_auc_score(targets, probabilities[:, 1])
                else:
                    metrics['auc'] = roc_auc_score(targets, probabilities.flatten())
            else:
                # Multi-class classification
                try:
                    metrics['auc_ovr'] = roc_auc_score(
                        targets, probabilities, multi_class='ovr', average=self.average
                    )
                    metrics['auc_ovo'] = roc_auc_score(
                        targets, probabilities, multi_class='ovo', average=self.average
                    )
                except ValueError:
                    # Handle case where not all classes are present
                    metrics['auc_ovr'] = 0.0
                    metrics['auc_ovo'] = 0.0
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.
        
        Returns:
            Confusion matrix as numpy array
        """
        if not self.all_predictions or not self.all_targets:
            raise ValueError("No predictions or targets available. Call update() first.")
        
        return confusion_matrix(
            self.all_targets,
            self.all_predictions,
            labels=list(range(self.num_classes))
        )
    
    def get_classification_report(self) -> str:
        """
        Get detailed classification report.
        
        Returns:
            Classification report as string
        """
        if not self.all_predictions or not self.all_targets:
            raise ValueError("No predictions or targets available. Call update() first.")
        
        return classification_report(
            self.all_targets,
            self.all_predictions,
            target_names=self.class_names,
            zero_division=0
        )
    
    def evaluate(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Dict[str, Union[float, np.ndarray, str]]:
        """
        Evaluate predictions against targets and return comprehensive results.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            probabilities: Class probabilities (optional)
            
        Returns:
            Dictionary containing all evaluation results
        """
        # Clear previous results
        self.reset()
        
        # Update with new data
        self.update(predictions, targets, probabilities)
        
        # Compute metrics
        metrics = self.compute_metrics()
        confusion_mat = self.get_confusion_matrix()
        classification_rep = self.get_classification_report()
        
        return {
            'metrics': metrics,
            'confusion_matrix': confusion_mat,
            'classification_report': classification_rep,
            'num_samples': len(self.all_targets),
            'class_distribution': self._get_class_distribution()
        }
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in targets."""
        from collections import Counter
        distribution = Counter(self.all_targets)
        return {
            self.class_names[i]: distribution.get(i, 0)
            for i in range(self.num_classes)
        }
    
    def reset(self):
        """Reset the evaluator state."""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
        self.metrics = ClassificationMetrics()
    
    def save_results(
        self,
        results: Dict[str, Union[float, np.ndarray, str]],
        output_path: str
    ):
        """
        Save evaluation results to file.
        
        Args:
            results: Results dictionary from evaluate()
            output_path: Path to save results
        """
        import json
        
        # Prepare serializable results
        serializable_results = {
            'metrics': results['metrics'],
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'classification_report': results['classification_report'],
            'num_samples': results['num_samples'],
            'class_distribution': results['class_distribution']
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Evaluation results saved to {output_path}")


class ModelEvaluator:
    """High-level evaluator for trained models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cpu',
        num_classes: int = 2,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on
            num_classes: Number of classes
            class_names: Names of the classes
        """
        self.model = model
        self.device = device
        self.evaluator = Evaluator(num_classes, class_names)
        
        self.model.to(device)
        self.model.eval()
    
    def evaluate_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Union[float, np.ndarray, str]]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            return_predictions: Whether to return individual predictions
            
        Returns:
            Evaluation results dictionary
        """
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Extract inputs and targets
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    segmentation = batch.get('segmentation', images).to(self.device)
                    targets = batch['labels'].to(self.device)
                else:
                    images, targets = batch[0].to(self.device), batch[1].to(self.device)
                    segmentation = images  # Fallback for non-dict batches
                
                # Ensure proper tensor dimensions for Conv3D (should be 5D: [batch, channels, depth, height, width])
                if images.dim() == 6:
                    images = images.squeeze(2)  # Remove extra dimension
                if segmentation.dim() == 6:
                    segmentation = segmentation.squeeze(2)  # Remove extra dimension
                
                # Ensure targets are 1D (batch_size,) for binary/multiclass classification
                if targets.dim() > 1:
                    targets = targets.squeeze()
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'segmentation' in self.model.forward.__code__.co_varnames:
                    # HF-compatible model
                    outputs = self.model(image=images, segmentation=segmentation)
                    logits = outputs.get('logits', outputs)
                else:
                    # Direct NodeDetector
                    outputs = self.model(images, segmentation)
                    logits = outputs
                
                # Get predictions and probabilities
                if logits.shape[1] == 1:
                    # Binary classification
                    probabilities = torch.sigmoid(logits)
                    predictions = (probabilities > 0.5).long().squeeze()
                    # Create 2-class probability matrix for consistency
                    probabilities = torch.cat([1 - probabilities, probabilities], dim=1)
                else:
                    # Multi-class classification
                    probabilities = torch.softmax(logits, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                all_probabilities.append(probabilities.cpu())
        
        # Handle empty dataloader case
        if not all_predictions:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0,
                'num_samples': 0
            }
        
        # Ensure all tensors have at least 1 dimension for concatenation
        all_predictions = [pred.view(-1) if pred.dim() == 0 else pred for pred in all_predictions]
        all_targets = [target.view(-1) if target.dim() == 0 else target for target in all_targets]
        all_probabilities = [prob if prob.dim() > 1 else prob.unsqueeze(0) for prob in all_probabilities]
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        all_probabilities = torch.cat(all_probabilities)
        
        # Evaluate
        results = self.evaluator.evaluate(
            all_predictions,
            all_targets,
            all_probabilities
        )
        
        if return_predictions:
            results['predictions'] = all_predictions.numpy()
            results['probabilities'] = all_probabilities.numpy()
        
        return results