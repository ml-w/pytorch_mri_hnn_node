import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import torch

class SegmentationVisualizer:
    """Interactive visualization tools for MRI lymph node segmentation validation."""
    
    def __init__(self, metrics: Dict[str, List[float]] = None):
        """Initialize visualizer with optional metrics history.
        
        Args:
            metrics: Dictionary of metric names to lists of values over time
        """
        self.metrics = metrics or {}
        self.figsize = (10, 6)
        
    def plot_metrics(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot training/validation metrics over time.
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        for metric, values in self.metrics.items():
            ax.plot(values, label=metric)
            
        ax.set_title('Training Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            
        return fig
        
    def plot_confusion_matrix(self, cm: np.ndarray, 
                            class_names: List[str] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix with annotations.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names)
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            
        return fig
        
    def show_sample(self, image: torch.Tensor, 
                   prediction: torch.Tensor,
                   ground_truth: torch.Tensor,
                   save_path: Optional[str] = None) -> plt.Figure:
        """Display sample image with prediction and ground truth.
        
        Args:
            image: Input image tensor (C, H, W)
            prediction: Model prediction tensor
            ground_truth: Ground truth tensor
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Input', 'Prediction', 'Ground Truth']
        
        for ax, data, title in zip(axes, [image, prediction, ground_truth], titles):
            ax.imshow(data.permute(1, 2, 0) if data.dim() == 3 else data)
            ax.set_title(title)
            ax.axis('off')
            
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            
        return fig

def launch_dashboard(metrics: Dict[str, List[float]], 
                    cm: np.ndarray,
                    samples: List[Dict]):
    """Launch interactive Streamlit dashboard.
    
    Args:
        metrics: Training metrics history
        cm: Confusion matrix
        samples: List of sample predictions
    """
    st.title('MRI Lymph Node Segmentation Validation')
    
    visualizer = SegmentationVisualizer(metrics)
    
    st.header('Training Metrics')
    fig = visualizer.plot_metrics()
    st.pyplot(fig)
    
    st.header('Confusion Matrix')
    fig = visualizer.plot_confusion_matrix(cm)
    st.pyplot(fig)
    
    st.header('Sample Predictions')
    sample_idx = st.selectbox('Select sample', range(len(samples)))
    sample = samples[sample_idx]
    fig = visualizer.show_sample(sample['image'], 
                               sample['prediction'],
                               sample['ground_truth'])
    st.pyplot(fig)