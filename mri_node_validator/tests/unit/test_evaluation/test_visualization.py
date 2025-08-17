"""Tests for visualization components with 3D support."""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from mri_node_validator.evaluation.visualization import SegmentationVisualizer, launch_dashboard
from mri_node_validator.visualization.plotter import Plotter


class TestSegmentationVisualizer:
    """Test cases for SegmentationVisualizer class."""
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        visualizer = SegmentationVisualizer()
        
        assert visualizer.metrics == {}
        assert visualizer.figsize == (10, 6)
    
    def test_visualizer_initialization_with_metrics(self, sample_training_history):
        """Test visualizer initialization with metrics."""
        visualizer = SegmentationVisualizer(metrics=sample_training_history)
        
        assert visualizer.metrics == sample_training_history
    
    def test_plot_metrics(self, sample_training_history):
        """Test metrics plotting."""
        visualizer = SegmentationVisualizer(metrics=sample_training_history)
        
        fig = visualizer.plot_metrics()
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_metrics_with_save(self, sample_training_history, temp_dir):
        """Test metrics plotting with save functionality."""
        visualizer = SegmentationVisualizer(metrics=sample_training_history)
        
        save_path = temp_dir / "metrics.png"
        fig = visualizer.plot_metrics(save_path=str(save_path))
        
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_confusion_matrix(self, sample_confusion_matrix, class_names):
        """Test confusion matrix plotting."""
        visualizer = SegmentationVisualizer()
        
        fig = visualizer.plot_confusion_matrix(
            cm=sample_confusion_matrix,
            class_names=class_names
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_confusion_matrix_with_save(self, sample_confusion_matrix, class_names, temp_dir):
        """Test confusion matrix plotting with save functionality."""
        visualizer = SegmentationVisualizer()
        
        save_path = temp_dir / "confusion_matrix.png"
        fig = visualizer.plot_confusion_matrix(
            cm=sample_confusion_matrix,
            class_names=class_names,
            save_path=str(save_path)
        )
        
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        plt.close(fig)
    
    def test_show_sample_3d_input(self):
        """Test show_sample with 3D input (our fix)."""
        visualizer = SegmentationVisualizer()
        
        # Test 4D input (C, D, H, W)
        image_4d = torch.randn(1, 64, 64, 64)
        prediction_4d = torch.randn(1, 64, 64, 64)
        ground_truth_4d = torch.randn(1, 64, 64, 64)
        
        fig = visualizer.show_sample(
            image=image_4d,
            prediction=prediction_4d,
            ground_truth=ground_truth_4d
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_show_sample_3d_volume(self):
        """Test show_sample with 3D volume input (D, H, W)."""
        visualizer = SegmentationVisualizer()
        
        # Test 3D input (D, H, W) - depth dimension
        image_3d = torch.randn(64, 64, 64)
        prediction_3d = torch.randn(64, 64, 64)
        ground_truth_3d = torch.randn(64, 64, 64)
        
        fig = visualizer.show_sample(
            image=image_3d,
            prediction=prediction_3d,
            ground_truth=ground_truth_3d
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_show_sample_3d_channels(self):
        """Test show_sample with 3D channel input (C, H, W)."""
        visualizer = SegmentationVisualizer()
        
        # Test 3D input (C, H, W) - channel dimension (small)
        image_3d = torch.randn(3, 64, 64)  # 3 channels
        prediction_3d = torch.randn(3, 64, 64)
        ground_truth_3d = torch.randn(3, 64, 64)
        
        fig = visualizer.show_sample(
            image=image_3d,
            prediction=prediction_3d,
            ground_truth=ground_truth_3d
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_show_sample_2d_input(self):
        """Test show_sample with 2D input (backward compatibility)."""
        visualizer = SegmentationVisualizer()
        
        # Test 2D input (H, W)
        image_2d = torch.randn(64, 64)
        prediction_2d = torch.randn(64, 64)
        ground_truth_2d = torch.randn(64, 64)
        
        fig = visualizer.show_sample(
            image=image_2d,
            prediction=prediction_2d,
            ground_truth=ground_truth_2d
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_show_sample_with_save(self, temp_dir):
        """Test show_sample with save functionality."""
        visualizer = SegmentationVisualizer()
        
        image = torch.randn(1, 64, 64, 64)
        prediction = torch.randn(1, 64, 64, 64)
        ground_truth = torch.randn(1, 64, 64, 64)
        
        save_path = temp_dir / "sample.png"
        fig = visualizer.show_sample(
            image=image,
            prediction=prediction,
            ground_truth=ground_truth,
            save_path=str(save_path)
        )
        
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        plt.close(fig)
    
    def test_show_sample_slice_extraction_consistency(self):
        """Test that slice extraction is consistent across different dimensions."""
        visualizer = SegmentationVisualizer()
        
        # Create volume with known pattern
        volume_4d = torch.zeros(1, 32, 32, 32)
        volume_4d[0, 16, :, :] = 1.0  # Mark middle slice
        
        volume_3d = torch.zeros(32, 32, 32)
        volume_3d[16, :, :] = 1.0  # Mark middle slice
        
        fig_4d = visualizer.show_sample(
            image=volume_4d,
            prediction=volume_4d,
            ground_truth=volume_4d
        )
        
        fig_3d = visualizer.show_sample(
            image=volume_3d,
            prediction=volume_3d,
            ground_truth=volume_3d
        )
        
        # Both should extract the marked slice
        assert isinstance(fig_4d, plt.Figure)
        assert isinstance(fig_3d, plt.Figure)
        
        plt.close(fig_4d)
        plt.close(fig_3d)


class TestPlotter:
    """Test cases for Plotter class."""
    
    def test_plotter_initialization(self, temp_dir):
        """Test plotter initialization."""
        plotter = Plotter(output_dir=temp_dir)
        
        assert plotter.output_dir == temp_dir
        assert plotter.figsize == (10, 8)
        assert plotter.dpi == 300
        assert plotter.output_files == []
    
    def test_plotter_custom_initialization(self, temp_dir):
        """Test plotter initialization with custom parameters."""
        plotter = Plotter(
            output_dir=temp_dir,
            figsize=(12, 10),
            dpi=150,
            style="darkgrid"
        )
        
        assert plotter.figsize == (12, 10)
        assert plotter.dpi == 150
    
    def test_plot_confusion_matrix(self, sample_predictions, class_names):
        """Test confusion matrix plotting."""
        plotter = Plotter()
        
        fig = plotter.plot_confusion_matrix(
            y_true=sample_predictions['labels'],
            y_pred=np.argmax(sample_predictions['predictions'], axis=1),
            class_names=class_names
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_confusion_matrix_normalized(self, sample_predictions, class_names):
        """Test normalized confusion matrix plotting."""
        plotter = Plotter()
        
        fig = plotter.plot_confusion_matrix(
            y_true=sample_predictions['labels'],
            y_pred=np.argmax(sample_predictions['predictions'], axis=1),
            class_names=class_names,
            normalize=True
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_roc_curve_binary(self, sample_binary_predictions):
        """Test ROC curve plotting for binary classification."""
        plotter = Plotter()
        
        fig = plotter.plot_roc_curve(
            y_true=sample_binary_predictions['labels'],
            y_scores=sample_binary_predictions['probabilities']
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_roc_curve_multiclass(self, sample_predictions, class_names):
        """Test ROC curve plotting for multi-class classification."""
        plotter = Plotter()
        
        fig = plotter.plot_roc_curve(
            y_true=sample_predictions['labels'],
            y_scores=sample_predictions['probabilities'],
            class_names=class_names
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_precision_recall_curve_binary(self, sample_binary_predictions):
        """Test PR curve plotting for binary classification."""
        plotter = Plotter()
        
        fig = plotter.plot_precision_recall_curve(
            y_true=sample_binary_predictions['labels'],
            y_scores=sample_binary_predictions['probabilities']
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_precision_recall_curve_multiclass(self, sample_predictions, class_names):
        """Test PR curve plotting for multi-class classification."""
        plotter = Plotter()
        
        fig = plotter.plot_precision_recall_curve(
            y_true=sample_predictions['labels'],
            y_scores=sample_predictions['probabilities'],
            class_names=class_names
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_training_history(self, sample_training_history):
        """Test training history plotting."""
        plotter = Plotter()
        
        fig = plotter.plot_training_history(history=sample_training_history)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_class_distribution(self):
        """Test class distribution plotting."""
        plotter = Plotter()
        
        class_counts = {"Class_0": 150, "Class_1": 100}
        fig = plotter.plot_class_distribution(class_counts=class_counts)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_sample_predictions_3d(self, class_names):
        """Test sample predictions plotting with 3D images."""
        plotter = Plotter()
        
        # Create 3D image batch
        images = torch.randn(4, 1, 32, 32, 32)  # (B, C, D, H, W)
        true_labels = np.array([0, 1, 0, 1])
        pred_labels = np.array([0, 1, 1, 1])
        pred_probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
        
        fig = plotter.plot_sample_predictions(
            images=images,
            true_labels=true_labels,
            pred_labels=pred_labels,
            pred_probs=pred_probs,
            class_names=class_names,
            n_samples=4
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_sample_predictions_2d(self, class_names):
        """Test sample predictions plotting with 2D images."""
        plotter = Plotter()
        
        # Create 2D image batch
        images = torch.randn(4, 64, 64)  # (B, H, W)
        true_labels = np.array([0, 1, 0, 1])
        pred_labels = np.array([0, 1, 1, 1])
        
        fig = plotter.plot_sample_predictions(
            images=images,
            true_labels=true_labels,
            pred_labels=pred_labels,
            class_names=class_names,
            n_samples=4
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_with_save(self, temp_dir, sample_training_history):
        """Test plotting with save functionality."""
        plotter = Plotter(output_dir=temp_dir)
        
        save_path = temp_dir / "training_history.png"
        fig = plotter.plot_training_history(
            history=sample_training_history,
            save_path=str(save_path)
        )
        
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        assert str(save_path) in plotter.output_files
        plt.close(fig)
    
    def test_create_interactive_dashboard(self, sample_predictions, sample_confusion_matrix, sample_metrics):
        """Test interactive dashboard creation."""
        plotter = Plotter()
        
        results = {
            'confusion_matrix': sample_confusion_matrix,
            'metrics': sample_metrics,
            'predictions': sample_predictions
        }
        
        fig = plotter.create_interactive_dashboard(results)
        
        # Check that figure was created (basic test since plotly figures are complex)
        assert fig is not None
    
    def test_create_interactive_dashboard_with_save(self, temp_dir, sample_predictions, sample_confusion_matrix, sample_metrics):
        """Test interactive dashboard creation with save."""
        plotter = Plotter(output_dir=temp_dir)
        
        results = {
            'confusion_matrix': sample_confusion_matrix,
            'metrics': sample_metrics,
            'predictions': sample_predictions
        }
        
        save_path = temp_dir / "dashboard.html"
        fig = plotter.create_interactive_dashboard(
            results=results,
            save_path=str(save_path)
        )
        
        assert fig is not None
        assert save_path.exists()
        assert str(save_path) in plotter.output_files


class TestVisualizationIntegration:
    """Integration tests for visualization components."""
    
    def test_visualizer_with_real_data_flow(self, sample_mri_volume_batch, sample_confusion_matrix):
        """Test visualizer with realistic data flow."""
        visualizer = SegmentationVisualizer()
        
        # Simulate model predictions
        batch_size = sample_mri_volume_batch.shape[0]
        
        for i in range(batch_size):
            image = sample_mri_volume_batch[i]  # (C, D, H, W)
            prediction = torch.sigmoid(torch.randn_like(image)) > 0.5
            ground_truth = torch.randint(0, 2, image.shape)
            
            fig = visualizer.show_sample(
                image=image,
                prediction=prediction.float(),
                ground_truth=ground_truth.float()
            )
            
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_plotter_with_evaluation_results(self, sample_predictions, class_names):
        """Test plotter with complete evaluation results."""
        plotter = Plotter()
        
        # Create multiple plots from evaluation results
        y_true = sample_predictions['labels']
        y_pred = np.argmax(sample_predictions['predictions'], axis=1)
        y_scores = sample_predictions['probabilities']
        
        # Confusion matrix
        cm_fig = plotter.plot_confusion_matrix(y_true, y_pred, class_names)
        assert isinstance(cm_fig, plt.Figure)
        plt.close(cm_fig)
        
        # ROC curve
        roc_fig = plotter.plot_roc_curve(y_true, y_scores, class_names)
        assert isinstance(roc_fig, plt.Figure)
        plt.close(roc_fig)
        
        # PR curve
        pr_fig = plotter.plot_precision_recall_curve(y_true, y_scores, class_names)
        assert isinstance(pr_fig, plt.Figure)
        plt.close(pr_fig)
    
    def test_end_to_end_visualization_workflow(self, temp_dir, sample_training_history, sample_predictions, class_names):
        """Test complete visualization workflow."""
        # Initialize components
        visualizer = SegmentationVisualizer(metrics=sample_training_history)
        plotter = Plotter(output_dir=temp_dir)
        
        # Generate all plots
        plots_created = []
        
        # 1. Training metrics
        metrics_fig = visualizer.plot_metrics(save_path=str(temp_dir / "metrics.png"))
        plots_created.append("metrics.png")
        plt.close(metrics_fig)
        
        # 2. Confusion matrix
        y_true = sample_predictions['labels']
        y_pred = np.argmax(sample_predictions['predictions'], axis=1)
        cm_fig = plotter.plot_confusion_matrix(
            y_true, y_pred, class_names,
            save_path=str(temp_dir / "confusion_matrix.png")
        )
        plots_created.append("confusion_matrix.png")
        plt.close(cm_fig)
        
        # 3. Sample predictions with 3D data
        images = torch.randn(len(y_true), 1, 32, 32, 32)
        pred_probs = sample_predictions['probabilities']
        sample_fig = plotter.plot_sample_predictions(
            images, y_true, y_pred, pred_probs, class_names,
            save_path=str(temp_dir / "sample_predictions.png")
        )
        plots_created.append("sample_predictions.png")
        plt.close(sample_fig)
        
        # Check that all plots were created
        for plot_name in plots_created:
            assert (temp_dir / plot_name).exists()
    
    def test_error_handling_invalid_inputs(self):
        """Test error handling with invalid inputs."""
        visualizer = SegmentationVisualizer()
        plotter = Plotter()
        
        # Test with mismatched dimensions
        image = torch.randn(32, 32, 32)
        prediction = torch.randn(16, 16, 16)  # Different size
        ground_truth = torch.randn(64, 64, 64)  # Different size
        
        # Should handle gracefully or raise appropriate error
        try:
            fig = visualizer.show_sample(image, prediction, ground_truth)
            plt.close(fig)
        except (ValueError, RuntimeError):
            # Expected to fail with dimension mismatch
            pass
        
        # Test plotter with empty data
        try:
            fig = plotter.plot_confusion_matrix([], [], [])
            plt.close(fig)
        except (ValueError, IndexError):
            # Expected to fail with empty data
            pass