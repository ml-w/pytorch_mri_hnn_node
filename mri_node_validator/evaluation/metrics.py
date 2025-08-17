from typing import Dict, List, Optional
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassAUROC
)

class ClassificationMetrics:
    """Handles computation and reporting of classification metrics."""
    
    def __init__(self, num_classes: int = 2, class_names: Optional[List[str]] = None):
        """Initialize metric collection.
        
        Args:
            num_classes: Number of classes for classification
            class_names: Optional list of class names for per-class metrics
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        
        if num_classes == 2:
            # Binary classification (regardless of class names)
            self.metrics = MetricCollection({
                'accuracy': BinaryAccuracy(),
                'precision': BinaryPrecision(),
                'recall': BinaryRecall(),
                'f1': BinaryF1Score(),
                'auroc': BinaryAUROC()
            })
        else:
            # Multi-class classification (3+ classes)
            self.metrics = MetricCollection({
                'accuracy': MulticlassAccuracy(num_classes=num_classes),
                'precision': MulticlassPrecision(num_classes=num_classes, average='macro'),
                'recall': MulticlassRecall(num_classes=num_classes, average='macro'),
                'f1': MulticlassF1Score(num_classes=num_classes, average='macro'),
                'auroc': MulticlassAUROC(num_classes=num_classes)
            })
            
            # Add per-class metrics for multiclass
            for i, class_name in enumerate(self.class_names):
                self.metrics[f'precision_{class_name}'] = MulticlassPrecision(
                    num_classes=num_classes, average=None
                )
                self.metrics[f'recall_{class_name}'] = MulticlassRecall(
                    num_classes=num_classes, average=None
                )
                self.metrics[f'f1_{class_name}'] = MulticlassF1Score(
                    num_classes=num_classes, average=None
                )
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new predictions and targets.
        
        Args:
            preds: Model predictions (probabilities)
            targets: Ground truth labels
        """
        self.metrics.update(preds, targets)
        
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all metrics.
        
        Returns:
            Dictionary of metric names to values
        """
        results = self.metrics.compute()
        
        # Handle per-class metrics for multiclass case (3+ classes only)
        if self.num_classes > 2:
            processed_results = {}
            for key, value in results.items():
                if any(class_name in key for class_name in self.class_names):
                    # This is a per-class metric - extract the specific class value
                    if value.numel() > 1:  # Multi-dimensional tensor
                        class_name = next(name for name in self.class_names if name in key)
                        class_idx = self.class_names.index(class_name)
                        processed_results[key] = value[class_idx]
                    else:
                        processed_results[key] = value
                else:
                    processed_results[key] = value
            return processed_results
        
        return results
        
    def generate_report(self, save_path: str = None) -> str:
        """Generate human-readable validation report with optional saving.
        
        Args:
            save_path: Optional path to save report (as .txt)
            
        Returns:
            Formatted string with metric results
        """
        results = self.compute()
        report = [
            "MRI Lymph Node Segmentation Validation Report",
            "============================================",
            f"Accuracy: {results['accuracy'].item():.4f}",
            f"Precision: {results['precision'].item():.4f}",
            f"Recall: {results['recall'].item():.4f}",
            f"F1 Score: {results['f1'].item():.4f}",
            f"AUROC: {results['auroc'].item():.4f}",
            "",
            "Confusion Matrix:",
            "----------------",
            "To be implemented in visualization module",
            ""
        ]
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
                
        return report_str
        
    def get_metrics_dict(self) -> Dict[str, float]:
        """Get metrics as a dictionary of float values.
        
        Returns:
            Dictionary of metric names to float values
        """
        results = self.compute()
        return {k: v.item() for k, v in results.items()}