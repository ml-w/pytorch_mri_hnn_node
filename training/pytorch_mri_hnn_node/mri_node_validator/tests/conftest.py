import pytest
import numpy as np
import torch

@pytest.fixture
def sample_mri_volume():
    """Fixture providing a sample 3D MRI volume"""
    return torch.randn(1, 64, 64, 64)  # (C, D, H, W)

@pytest.fixture
def sample_segmentation_mask():
    """Fixture providing a sample segmentation mask"""
    mask = torch.zeros(1, 64, 64, 64)
    mask[:, 10:20, 10:20, 10:20] = 1  # Single node
    mask[:, 30:40, 30:40, 30:40] = 2  # Another node
    return mask

@pytest.fixture
def merged_node_mask():
    """Fixture providing a mask with merged nodes"""
    mask = torch.zeros(1, 64, 64, 64)
    mask[:, 10:30, 10:30, 10:30] = 1  # Merged nodes
    return mask

@pytest.fixture
def sample_bounding_boxes():
    """Fixture providing sample bounding boxes in JSON format"""
    return [
        {"label": 1, "coordinates": [10, 10, 10, 20, 20, 20]},
        {"label": 2, "coordinates": [30, 30, 30, 40, 40, 40]}
    ]