"""Plotter module for MRI classification visualization and reporting."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class Plotter:
    """Main plotter class for MRI classification visualization."""
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "results",
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 300,
        style: str = "whitegrid"
    ):
        """
        Initialize the plotter.
        
        Args:
            output_dir: Directory to save plots
            figsize: Default figure size
            dpi: DPI for saved figures
            style: Seaborn style
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        self.output_files = []
        
        # Set style
        sns.set_style(style)
        plt.rcParams['figure.dpi'] = dpi
    
    def plot_confusion_matrix(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        class_names: Optional[List[str]] = None,
        normalize: bool = False,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            normalize: Whether to normalize the matrix
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.output_files.append(save_path)
        
        return fig
    
    def plot_roc_curve(
        self,
        y_true: Union[np.ndarray, List],
        y_scores: Union[np.ndarray, List],
        class_names: Optional[List[str]] = None,
        title: str = "ROC Curve",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curve for binary or multi-class classification.
        
        Args:
            y_true: True labels
            y_scores: Prediction scores/probabilities
            class_names: Names of classes
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        y_scores = np.array(y_scores)
        y_true = np.array(y_true)
        
        if y_scores.ndim == 1 or y_scores.shape[1] == 1:
            # Binary classification
            y_scores_flat = y_scores.flatten()
            fpr, tpr, _ = roc_curve(y_true, y_scores_flat)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            # Multi-class classification
            n_classes = y_scores.shape[1]
            
            # For binary classification with 2D output (n_samples, 2),
            # we need to handle it differently than multi-class
            if n_classes == 2:
                # Binary classification with 2D output - use positive class probabilities
                y_scores_pos = y_scores[:, 1]  # Positive class probabilities
                fpr, tpr, _ = roc_curve(y_true, y_scores_pos)
                roc_auc = auc(fpr, tpr)
                class_name = class_names[1] if class_names and len(class_names) > 1 else 'Class 1'
                ax.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
            else:
                # True multi-class classification
                y_true_bin = label_binarize(y_true, classes=range(n_classes))
                
                # Plot ROC curve for each class
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                    roc_auc = auc(fpr, tpr)
                    class_name = class_names[i] if class_names else f'Class {i}'
                    ax.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.output_files.append(save_path)
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        y_true: Union[np.ndarray, List],
        y_scores: Union[np.ndarray, List],
        class_names: Optional[List[str]] = None,
        title: str = "Precision-Recall Curve",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_scores: Prediction scores/probabilities
            class_names: Names of classes
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        y_scores = np.array(y_scores)
        y_true = np.array(y_true)
        
        if y_scores.ndim == 1 or y_scores.shape[1] == 1:
            # Binary classification
            y_scores_flat = y_scores.flatten()
            precision, recall, _ = precision_recall_curve(y_true, y_scores_flat)
            pr_auc = auc(recall, precision)
            
            ax.plot(recall, precision, linewidth=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        else:
            # Multi-class classification
            n_classes = y_scores.shape[1]
            
            # For binary classification with 2D output (n_samples, 2),
            # we need to handle it differently than multi-class
            if n_classes == 2:
                # Binary classification with 2D output - use positive class probabilities
                y_scores_pos = y_scores[:, 1]  # Positive class probabilities
                precision, recall, _ = precision_recall_curve(y_true, y_scores_pos)
                pr_auc = auc(recall, precision)
                class_name = class_names[1] if class_names and len(class_names) > 1 else 'Class 1'
                ax.plot(recall, precision, linewidth=2, label=f'{class_name} (AUC = {pr_auc:.2f})')
            else:
                # True multi-class classification
                y_true_bin = label_binarize(y_true, classes=range(n_classes))
                
                for i in range(n_classes):
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
                    pr_auc = auc(recall, precision)
                    class_name = class_names[i] if class_names else f'Class {i}'
                    ax.plot(recall, precision, linewidth=2, label=f'{class_name} (AUC = {pr_auc:.2f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.output_files.append(save_path)
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training history (loss, metrics over epochs).
        
        Args:
            history: Dictionary with metric names as keys and lists of values
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Separate loss and other metrics
        loss_metrics = {k: v for k, v in history.items() if 'loss' in k.lower()}
        other_metrics = {k: v for k, v in history.items() if 'loss' not in k.lower()}
        
        # Determine number of subplots
        n_plots = 1 if loss_metrics else 0
        n_plots += 1 if other_metrics else 0
        
        if n_plots == 0:
            raise ValueError("No metrics found in history")
        
        fig, axes = plt.subplots(1, n_plots, figsize=(self.figsize[0] * n_plots, self.figsize[1]))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot loss metrics
        if loss_metrics:
            ax = axes[plot_idx]
            for metric_name, values in loss_metrics.items():
                ax.plot(values, label=metric_name, linewidth=2)
            ax.set_title('Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot other metrics
        if other_metrics:
            ax = axes[plot_idx]
            for metric_name, values in other_metrics.items():
                ax.plot(values, label=metric_name, linewidth=2)
            ax.set_title('Metrics')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.output_files.append(save_path)
        
        return fig
    
    def plot_class_distribution(
        self,
        class_counts: Dict[str, int],
        title: str = "Class Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot class distribution as bar chart.
        
        Args:
            class_counts: Dictionary with class names and counts
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = ax.bar(classes, counts, alpha=0.8)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                   f'{count}', ha='center', va='bottom')
        
        ax.set_title(title)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.output_files.append(save_path)
        
        return fig
    
    def plot_sample_predictions(
        self,
        images: Union[torch.Tensor, np.ndarray],
        true_labels: Union[np.ndarray, List],
        pred_labels: Union[np.ndarray, List],
        pred_probs: Optional[Union[np.ndarray, List]] = None,
        class_names: Optional[List[str]] = None,
        n_samples: int = 8,
        title: str = "Sample Predictions",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot sample images with predictions.
        
        Args:
            images: Input images
            true_labels: True labels
            pred_labels: Predicted labels
            pred_probs: Prediction probabilities
            class_names: Names of classes
            n_samples: Number of samples to plot
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        n_samples = min(n_samples, len(images))
        n_cols = min(4, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_samples):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Get middle slice for 3D images
            if images[i].ndim == 4:  # (C, D, H, W)
                img_slice = images[i][0, images[i].shape[1]//2, :, :]
            elif images[i].ndim == 3:  # (D, H, W)
                img_slice = images[i][images[i].shape[0]//2, :, :]
            else:  # 2D image
                img_slice = images[i]
            
            ax.imshow(img_slice, cmap='gray')
            
            # Create title with prediction info
            true_class = class_names[true_labels[i]] if class_names else f"Class {true_labels[i]}"
            pred_class = class_names[pred_labels[i]] if class_names else f"Class {pred_labels[i]}"
            
            title_text = f"True: {true_class}\nPred: {pred_class}"
            if pred_probs is not None:
                confidence = pred_probs[i][pred_labels[i]] if pred_probs[i].ndim > 0 else pred_probs[i]
                title_text += f"\nConf: {confidence:.2f}"
            
            # Color title based on correctness
            color = 'green' if true_labels[i] == pred_labels[i] else 'red'
            ax.set_title(title_text, color=color, fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_samples, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.output_files.append(save_path)
        
        return fig
    
    def generate_plots(
        self,
        image: torch.Tensor,
        segmentation: torch.Tensor,
        predictions: Dict[str, Any],
        metrics: Dict[str, float]
    ):
        """
        Generate all plots for validation results.
        
        Args:
            image: Input image tensor
            segmentation: Segmentation tensor
            predictions: Prediction results
            metrics: Computed metrics
        """
        # This method is kept for backward compatibility with the existing validator
        # In practice, you would call the specific plotting methods above
        pass
    
    def create_interactive_dashboard(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            results: Results dictionary containing metrics, predictions, etc.
            save_path: Path to save the HTML dashboard
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curve', 'Metrics', 'Class Distribution'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add confusion matrix
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            fig.add_trace(
                go.Heatmap(z=cm, colorscale='Blues', showscale=False),
                row=1, col=1
            )
        
        # Add metrics bar chart
        if 'metrics' in results:
            metrics = results['metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            fig.add_trace(
                go.Bar(x=metric_names, y=metric_values, name='Metrics'),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="MRI Classification Results Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            self.output_files.append(save_path)
        
        return fig