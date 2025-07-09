from typing import Dict, List
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC
)

class ClassificationMetrics:
    """Handles computation and reporting of classification metrics."""
    
    def __init__(self):
        """Initialize metric collection."""
        self.metrics = MetricCollection({
            'accuracy': BinaryAccuracy(),
            'precision': BinaryPrecision(),
            'recall': BinaryRecall(),
            'f1': BinaryF1Score(),
            'auroc': BinaryAUROC()
        })
        
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
        return self.metrics.compute()
        
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