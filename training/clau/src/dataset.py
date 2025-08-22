import torch
import torchio as tio

def permute_and_squeeze(x: torch.Tensor) -> torch.Tensor:
    return x.permute(3, 1, 2, 0).squeeze(-1)

def get_transforms(train: bool):
    """
    Defines the preprocessing and augmentation pipeline using Torchio.
    """

    if train:
        # Training pipeline: includes augmentations
        transform = tio.Compose([
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.RandomFlip(axes=('LR',), flip_probability=0.5),
            tio.RandomFlip(axes=('AP',), flip_probability=0.5),
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
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        ])
    return transform