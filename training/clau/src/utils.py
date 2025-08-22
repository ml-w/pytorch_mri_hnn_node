import torch
import numpy as np
import random
import os

def seed_everything(seed: int = 42):
    """
    Seeds all relevant random number generators for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Configuration Dictionary ---
CONFIG = {
    "data": {
        "root_dir": "training/clau/data",
        "labels_csv": "training/clau/data/merged_small/merged_small.csv",
        "train_folds": [0, 1, 2, 3],
        "val_folds": [4],
    },
    "model": {
        "name": "resnet50",
        "pretrained": True,
        "freeze_epochs": 2, # Number of epochs where only the final layer is trained
    },
    "training": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "batch_size": 16, # Adjust based on your GPU memory
        "num_epochs": 15,
        "num_workers": os.cpu_count() // 2,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "use_amp": True, # Use Automatic Mixed Precision
        "patience": 5,   # Early stopping patience
        "learning_rate": 1e-4,
        "scheduler_mode": "max",
        "scheduler_patience": 3,
        "scheduler_factor": 0.1,
        "n_splits": 5,
    },
    "output": {
        "dir": "training/clau/outputs",
    }
}