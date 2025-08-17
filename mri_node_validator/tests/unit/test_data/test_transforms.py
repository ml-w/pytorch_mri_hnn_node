"""Tests for 3D MRI transforms and preprocessing."""

import pytest
import torch
import torchio as tio
import numpy as np

from mri_node_validator.data.transforms import (
    get_default_transforms,
    get_training_transforms,
    get_validation_transforms,
    get_inference_transforms,
    get_preprocessing_pipeline,
    CustomTransforms
)


class TestDefaultTransforms:
    """Test cases for default transform functions."""
    
    def test_get_default_transforms_basic(self):
        """Test basic default transforms creation."""
        transforms = get_default_transforms()
        
        assert isinstance(transforms, tio.Compose)
        assert len(transforms.transforms) > 0
    
    def test_get_default_transforms_with_resampling(self):
        """Test default transforms with custom spacing."""
        target_spacing = (2.0, 2.0, 2.0)
        transforms = get_default_transforms(target_spacing=target_spacing)
        
        # Check that Resample transform is included
        has_resample = any(isinstance(t, tio.Resample) for t in transforms.transforms)
        assert has_resample
    
    def test_get_default_transforms_with_cropping(self):
        """Test default transforms with target shape."""
        target_shape = (64, 64, 64)
        transforms = get_default_transforms(target_shape=target_shape)
        
        # Check that CropOrPad transform is included
        has_crop_pad = any(isinstance(t, tio.CropOrPad) for t in transforms.transforms)
        assert has_crop_pad
    
    def test_get_default_transforms_with_normalization(self):
        """Test default transforms with normalization."""
        transforms = get_default_transforms(normalize=True)
        
        # Check that ZNormalization is included
        has_znorm = any(isinstance(t, tio.ZNormalization) for t in transforms.transforms)
        assert has_znorm
    
    def test_get_default_transforms_with_augmentation(self):
        """Test default transforms with augmentation."""
        transforms = get_default_transforms(augment=True, augmentation_probability=0.5)
        
        # Check that augmentation transforms are included
        transform_types = [type(t) for t in transforms.transforms]
        
        # Should have at least some augmentation transforms
        aug_transforms = [tio.RandomAffine, tio.RandomNoise, tio.RandomBlur, 
                         tio.RandomBiasField, tio.RandomGamma, tio.RandomFlip]
        has_augmentation = any(aug_type in transform_types for aug_type in aug_transforms)
        assert has_augmentation
    
    def test_get_default_transforms_no_augmentation(self):
        """Test default transforms without augmentation."""
        transforms = get_default_transforms(augment=False)
        
        # Should not have augmentation transforms
        transform_types = [type(t) for t in transforms.transforms]
        aug_transforms = [tio.RandomAffine, tio.RandomNoise, tio.RandomBlur, 
                         tio.RandomBiasField, tio.RandomGamma, tio.RandomFlip]
        has_augmentation = any(aug_type in transform_types for aug_type in aug_transforms)
        assert not has_augmentation


class TestSpecificTransforms:
    """Test cases for specific transform functions."""
    
    def test_get_training_transforms(self):
        """Test training transforms creation."""
        transforms = get_training_transforms()
        
        assert isinstance(transforms, tio.Compose)
        
        # Training should include augmentation
        transform_types = [type(t) for t in transforms.transforms]
        aug_transforms = [tio.RandomAffine, tio.RandomNoise]
        has_augmentation = any(aug_type in transform_types for aug_type in aug_transforms)
        assert has_augmentation
    
    def test_get_validation_transforms(self):
        """Test validation transforms creation."""
        transforms = get_validation_transforms()
        
        assert isinstance(transforms, tio.Compose)
        
        # Validation should not include augmentation
        transform_types = [type(t) for t in transforms.transforms]
        aug_transforms = [tio.RandomAffine, tio.RandomNoise, tio.RandomBlur]
        has_augmentation = any(aug_type in transform_types for aug_type in aug_transforms)
        assert not has_augmentation
    
    def test_get_inference_transforms(self):
        """Test inference transforms creation."""
        transforms = get_inference_transforms()
        
        assert isinstance(transforms, tio.Compose)
        
        # Inference should not include augmentation
        transform_types = [type(t) for t in transforms.transforms]
        aug_transforms = [tio.RandomAffine, tio.RandomNoise, tio.RandomBlur]
        has_augmentation = any(aug_type in transform_types for aug_type in aug_transforms)
        assert not has_augmentation
    
    def test_training_vs_validation_transforms(self):
        """Test difference between training and validation transforms."""
        train_transforms = get_training_transforms()
        val_transforms = get_validation_transforms()
        
        # Training should have more transforms due to augmentation
        assert len(train_transforms.transforms) > len(val_transforms.transforms)


class TestTransformApplication:
    """Test cases for transform application on 3D data."""
    
    def test_resample_transform(self, sample_mri_volume):
        """Test resampling transform on 3D volume."""
        import numpy as np
        
        # Create subject with proper spacing using affine matrix
        affine = np.eye(4) * 2.0  # 2mm spacing
        affine[3, 3] = 1.0  # Keep homogeneous coordinate as 1
        
        subject = tio.Subject({
            'image': tio.ScalarImage(tensor=sample_mri_volume, affine=affine)
        })
        
        transform = tio.Resample((1.0, 1.0, 1.0))
        transformed = transform(subject)
        
        # Should approximately double the size in each dimension
        original_shape = subject['image'].shape[1:]  # Skip channel dimension
        new_shape = transformed['image'].shape[1:]
        
        # Allow for some rounding differences
        for orig, new in zip(original_shape, new_shape):
            assert abs(new - orig * 2) <= 2
    
    def test_crop_or_pad_transform(self, sample_mri_volume):
        """Test CropOrPad transform."""
        subject = tio.Subject({'image': tio.ScalarImage(tensor=sample_mri_volume)})
        
        target_shape = (32, 32, 32)
        transform = tio.CropOrPad(target_shape)
        transformed = transform(subject)
        
        assert transformed['image'].shape[1:] == target_shape
    
    def test_znormalization_transform(self, sample_mri_volume):
        """Test Z-normalization transform."""
        subject = tio.Subject({'image': tio.ScalarImage(tensor=sample_mri_volume)})
        
        transform = tio.ZNormalization()
        transformed = transform(subject)
        
        # Check that mean is approximately 0 and std is approximately 1
        data = transformed['image'].data
        assert abs(data.mean().item()) < 0.1
        assert abs(data.std().item() - 1.0) < 0.1
    
    def test_random_affine_transform(self, sample_mri_volume):
        """Test RandomAffine transform."""
        subject = tio.Subject({'image': tio.ScalarImage(tensor=sample_mri_volume)})
        
        transform = tio.RandomAffine(
            scales=(0.9, 1.1),
            degrees=(-10, 10),
            translation=(-5, 5),
            p=1.0  # Always apply
        )
        
        # Apply multiple times to check randomness
        transformed1 = transform(subject)
        transformed2 = transform(subject)
        
        # Results should be different (with high probability)
        diff = torch.abs(transformed1['image'].data - transformed2['image'].data).sum()
        assert diff > 0
    
    def test_random_noise_transform(self, sample_mri_volume):
        """Test RandomNoise transform."""
        subject = tio.Subject({'image': tio.ScalarImage(tensor=sample_mri_volume)})
        
        transform = tio.RandomNoise(std=(0, 0.1), p=1.0)
        transformed = transform(subject)
        
        # Should have different values due to noise
        diff = torch.abs(transformed['image'].data - subject['image'].data).sum()
        assert diff > 0
    
    def test_random_flip_transform(self, sample_mri_volume):
        """Test RandomFlip transform."""
        subject = tio.Subject({'image': tio.ScalarImage(tensor=sample_mri_volume)})
        
        transform = tio.RandomFlip(axes=('LR',), p=1.0)
        transformed = transform(subject)
        
        # Shape should remain the same
        assert transformed['image'].shape == subject['image'].shape
    
    def test_transform_pipeline(self, sample_mri_volume):
        """Test complete transform pipeline."""
        subject = tio.Subject({'image': tio.ScalarImage(tensor=sample_mri_volume)})
        
        transforms = get_training_transforms(
            target_spacing=(1.0, 1.0, 1.0),
            target_shape=(32, 32, 32),
            augmentation_probability=0.5
        )
        
        transformed = transforms(subject)
        
        # Check final shape
        assert transformed['image'].shape[1:] == (32, 32, 32)
        
        # Check that data is still a valid tensor
        assert torch.is_tensor(transformed['image'].data)
        assert not torch.any(torch.isnan(transformed['image'].data))


class TestCustomTransforms:
    """Test cases for custom transform implementations."""
    
    def test_intensity_clipping(self, sample_mri_volume):
        """Test intensity clipping transform."""
        # Create volume with known outliers
        volume = torch.randn(1, 32, 32, 32)
        volume[0, 0, 0, 0] = 100.0  # Outlier
        volume[0, 0, 0, 1] = -100.0  # Outlier
        
        transform = CustomTransforms.intensity_clipping(
            percentile_low=5.0,
            percentile_high=95.0
        )
        
        clipped = transform(volume)
        
        # Outliers should be clipped
        assert clipped.max() < 100.0
        assert clipped.min() > -100.0
    
    def test_brain_extraction_mask(self, sample_mri_volume):
        """Test brain extraction transform."""
        # Create volume with background and foreground
        volume = torch.zeros(1, 32, 32, 32)
        volume[0, 10:20, 10:20, 10:20] = 1.0  # Brain region
        
        transform = CustomTransforms.brain_extraction_mask()
        masked = transform(volume)
        
        # Background should remain zero
        assert masked[0, 0, 0, 0].item() == 0.0
        
        # Brain region should be preserved or modified
        assert masked[0, 15, 15, 15].item() >= 0.0


class TestPreprocessingPipeline:
    """Test cases for preprocessing pipeline creation."""
    
    def test_get_preprocessing_pipeline_basic(self):
        """Test basic preprocessing pipeline creation."""
        config = {
            'target_spacing': [1.0, 1.0, 1.0],
            'target_shape': [64, 64, 64],
            'augmentation_probability': 0.5
        }
        
        train_transforms, val_transforms, test_transforms = get_preprocessing_pipeline(config)
        
        assert isinstance(train_transforms, tio.Compose)
        assert isinstance(val_transforms, tio.Compose)
        assert isinstance(test_transforms, tio.Compose)
        
        # Training should have more transforms (augmentation)
        assert len(train_transforms.transforms) > len(val_transforms.transforms)
        assert len(val_transforms.transforms) == len(test_transforms.transforms)
    
    def test_get_preprocessing_pipeline_with_custom_options(self):
        """Test preprocessing pipeline with custom options."""
        config = {
            'target_spacing': [2.0, 2.0, 2.0],
            'target_shape': [128, 128, 128],
            'augmentation_probability': 0.8,
            'intensity_clipping': True,
            'clipping_percentiles': [2.0, 98.0],
            'brain_extraction': True
        }
        
        train_transforms, val_transforms, test_transforms = get_preprocessing_pipeline(config)
        
        # Check that custom transforms are included
        train_transform_types = [type(t) for t in train_transforms.transforms]
        
        # Should have resample and crop/pad
        assert tio.Resample in train_transform_types
        assert tio.CropOrPad in train_transform_types
        assert tio.ZNormalization in train_transform_types
    
    def test_preprocessing_pipeline_reproducibility(self, sample_mri_volume):
        """Test that preprocessing pipeline is reproducible."""
        config = {
            'target_spacing': [1.0, 1.0, 1.0],
            'target_shape': [32, 32, 32],
            'augmentation_probability': 0.0  # No randomness
        }
        
        _, val_transforms, _ = get_preprocessing_pipeline(config)
        
        subject = tio.Subject({'image': tio.ScalarImage(tensor=sample_mri_volume)})
        
        # Apply transforms multiple times
        result1 = val_transforms(subject)
        result2 = val_transforms(subject)
        
        # Results should be identical (no augmentation)
        assert torch.allclose(result1['image'].data, result2['image'].data, atol=1e-6)
    
    def test_preprocessing_pipeline_with_missing_config(self):
        """Test preprocessing pipeline with minimal config."""
        config = {}  # Empty config
        
        train_transforms, val_transforms, test_transforms = get_preprocessing_pipeline(config)
        
        # Should use default values
        assert isinstance(train_transforms, tio.Compose)
        assert isinstance(val_transforms, tio.Compose)
        assert isinstance(test_transforms, tio.Compose)
        
        # Should have at least some transforms
        assert len(train_transforms.transforms) > 0
        assert len(val_transforms.transforms) > 0