#!/usr/bin/env python3
"""
Minimal Example: Download and use a pre-trained MRI node classification model

This script demonstrates the simplest way to:
1. Download a pre-trained model from Hugging Face Hub
2. Run inference on a single MRI volume
"""

import torch
import numpy as np
from mri_node_validator.models.hf_detector import HFNodeDetector

def main():
    # Example 1: Load a pre-trained model from Hugging Face Hub
    # Note: Replace with actual model name when available on HF Hub
    try:
        # This would be the actual usage when model is published
        # model = HFNodeDetector.from_pretrained("huggingface-user/mri-node-detector")
        
        # For now, we'll create a model and show how to save/load locally
        print("Creating and saving a model locally (demo)...")
        from mri_node_validator.models.hf_detector import MRIConfig
        
        # Create model configuration
        config = MRIConfig(
            learning_rate=1e-3,
            architecture_type="3d_resnet_attention",
            num_labels=2  # Binary classification: merged vs. non-merged nodes
        )
        
        # Initialize model
        model = HFNodeDetector(config)
        
        # Save model locally (in real usage, this would be downloaded from HF Hub)
        model.save_pretrained("./demo_mri_model")
        print("✓ Model saved to ./demo_mri_model")
        
        # Load the model (simulates downloading from HF Hub)
        model = HFNodeDetector.from_pretrained("./demo_mri_model")
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Example 2: Create sample MRI data (in real usage, load your NIfTI files)
    batch_size = 1
    channels = 1
    depth, height, width = 64, 64, 64
    
    # Create dummy MRI volume and segmentation mask
    mri_volume = torch.randn(batch_size, channels, depth, height, width)
    segmentation_mask = torch.randint(0, 2, (batch_size, channels, depth, height, width)).float()
    
    print(f"Input shapes - MRI: {mri_volume.shape}, Segmentation: {segmentation_mask.shape}")

    # Example 3: Run inference
    model.eval()
    with torch.no_grad():
        # Forward pass
        outputs = model(image=mri_volume, segmentation=segmentation_mask)
        
        # Get predictions
        logits = outputs['logits']
        probabilities = torch.sigmoid(logits)  # For binary classification
        predicted_class = (probabilities > 0.5).long()
        
        print(f"Raw logits: {logits}")
        print(f"Probabilities: {probabilities}")
        print(f"Predicted class: {predicted_class.item()}")
        print(f"Confidence: {probabilities.item():.3f}")
        
        # Interpret results
        class_names = ["No Merged Nodes", "Merged Nodes Detected"]
        prediction = class_names[predicted_class.item()]
        confidence = probabilities.item() if predicted_class.item() == 1 else (1 - probabilities.item())
        
        print(f"\n🏥 Diagnosis: {prediction}")
        print(f"📊 Confidence: {confidence:.1%}")

if __name__ == "__main__":
    main()