import torch
import pandas as pd
import os
import torchio as tio

class NiftiDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for loading 3-channel NIfTI patches.

    Args:
        df (pd.DataFrame): DataFrame with 'filename' and 'individual' columns.
        root_dir (str): The root directory where NIfTI files are stored.
        transform (callable, optional): A Torchio transform pipeline. Defaults to None.
    """
    def __init__(self, df: pd.DataFrame, root_dir: str, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = os.path.join(self.root_dir, row['filename'])
        individual = torch.tensor(row['individual'], dtype=torch.float)

        # A Torchio Subject is a dictionary of images.
        # We load our 3-slice NIfTI as a single ScalarImage.
        # The shape (224, 224, 3) will be interpreted as (W, H, D) where D is the channel dim.
        subject = tio.Subject(
            mri=tio.ScalarImage(filename)
        )

        # Apply transforms if any
        if self.transform:
            subject = self.transform(subject)
        
        # Get the tensor from the transformed subject.
        # Torchio handles the channel permutation to (C, H, W)
        image_tensor = subject.mri.data.float()

        return image_tensor, individual

def get_transforms(train: bool):
    """
    Defines the preprocessing and augmentation pipeline using Torchio.
    
    Args:
        train (bool): If True, returns the training pipeline with augmentations.
                      Otherwise, returns the validation/test pipeline.
    """
    if train:
        # Training pipeline: includes augmentations
        transform = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1)), # Scale intensities to [0, 1]
            tio.ZNormalization(masking_method=tio.ZNormalization.mean), # Per-volume z-score
            tio.RandomFlip(axes=('LR',), flip_probability=0.5), # Horizontal Flip
            tio.RandomFlip(axes=('AP',), flip_probability=0.5), # Vertical Flip
            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=10,
                translation=5,
                isotropic=True,
                default_pad_value='mean'
            ),
            tio.RandomGamma(log_gamma=(-0.2, 0.2)),
        ])
    else:
        # Validation/Test pipeline: only preprocessing
        transform = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1)),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        ])
    return transform