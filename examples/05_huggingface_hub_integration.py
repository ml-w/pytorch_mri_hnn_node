#!/usr/bin/env python3
"""
Hugging Face Hub Integration Example

This script demonstrates:
1. Uploading trained models to Hugging Face Hub
2. Downloading models from Hugging Face Hub
3. Model versioning and metadata management
4. Sharing models with the community
"""

import os
import torch
import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from huggingface_hub import HfApi, Repository, login, logout
from mri_node_validator.models.hf_detector import HFNodeDetector, MRIConfig

def setup_huggingface_auth():
    """Setup Hugging Face authentication."""
    
    print("🔐 Hugging Face Authentication Setup")
    print("=" * 40)
    
    # Check if already logged in
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✓ Already logged in as: {user['name']}")
        return True
    except Exception:
        pass
    
    print("\nTo upload models to Hugging Face Hub, you need to:")
    print("1. Create an account at https://huggingface.co")
    print("2. Create an access token at https://huggingface.co/settings/tokens")
    print("3. Run: huggingface-cli login")
    print("   OR")
    print("4. Set HF_TOKEN environment variable")
    
    # Try to login with environment variable
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        try:
            login(token=hf_token)
            user = api.whoami()
            print(f"✓ Logged in with token as: {user['name']}")
            return True
        except Exception as e:
            print(f"❌ Login failed: {e}")
    
    print("\n⚠️  Skipping upload examples (authentication required)")
    return False

def create_model_card(model_config: MRIConfig, 
                     training_info: Dict,
                     performance_metrics: Dict) -> str:
    """Create a model card for Hugging Face Hub."""
    
    model_card = f"""---
language: en
tags:
- medical-imaging
- mri
- lymph-node
- classification
- 3d-cnn
- pytorch
- huggingface
license: apache-2.0
datasets:
- custom
metrics:
- accuracy
- f1
- precision
- recall
- auc
---

# MRI Lymph Node Classification Model

## Model Description

This model performs binary classification on 3D MRI volumes to detect merged lymph nodes. It uses a 3D ResNet architecture with attention mechanisms specifically designed for medical imaging tasks.

## Model Details

- **Model Type**: 3D Convolutional Neural Network
- **Architecture**: {model_config.architecture_type}
- **Task**: Binary Classification (Merged vs. Non-merged lymph nodes)
- **Input**: 3D MRI volumes + segmentation masks
- **Output**: Binary classification probabilities

## Training Data

- **Dataset Size**: {training_info.get('num_samples', 'N/A')} samples
- **Input Shape**: {training_info.get('input_shape', '[64, 64, 64]')}
- **Augmentation**: {training_info.get('augmentation', 'Standard medical imaging augmentations')}

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | {performance_metrics.get('accuracy', 0):.4f} |
| Precision | {performance_metrics.get('precision', 0):.4f} |
| Recall | {performance_metrics.get('recall', 0):.4f} |
| F1 Score | {performance_metrics.get('f1', 0):.4f} |
| AUC | {performance_metrics.get('auc', 0):.4f} |

## Usage

```python
from mri_node_validator.models.hf_detector import HFNodeDetector
import torch

# Load the model
model = HFNodeDetector.from_pretrained("your-username/mri-node-classifier")

# Prepare your data (example)
mri_volume = torch.randn(1, 1, 64, 64, 64)  # Batch, Channel, Depth, Height, Width
segmentation = torch.randn(1, 1, 64, 64, 64)

# Run inference
model.eval()
with torch.no_grad():
    outputs = model(image=mri_volume, segmentation=segmentation)
    probabilities = torch.sigmoid(outputs['logits'])
    prediction = (probabilities > 0.5).long()

print(f"Prediction: {{prediction.item()}}")
print(f"Confidence: {{probabilities.item():.3f}}")
```

## Training Procedure

### Training Hyperparameters

- **Learning Rate**: {model_config.learning_rate}
- **Batch Size**: {training_info.get('batch_size', 'N/A')}
- **Epochs**: {training_info.get('num_epochs', 'N/A')}
- **Optimizer**: Adam
- **Loss Function**: Binary Cross Entropy

### Data Preprocessing

1. Intensity normalization to [0, 1] range
2. Resampling to {training_info.get('target_spacing', '[1.0, 1.0, 1.0]')} mm spacing
3. Cropping/padding to {training_info.get('input_shape', '[64, 64, 64]')} voxels
4. Data augmentation (rotation, flipping, noise injection)

## Limitations and Bias

- This model was trained on a specific dataset and may not generalize to all MRI scanners or protocols
- Performance may vary with different image quality or acquisition parameters
- The model should not be used as the sole diagnostic tool
- Always consult with medical professionals for clinical decisions

## Citation

If you use this model, please cite:

```bibtex
@misc{{mri-node-classifier,
  title={{MRI Lymph Node Classification Model}},
  author={{Your Name}},
  year={{2024}},
  howpublished={{\\url{{https://huggingface.co/your-username/mri-node-classifier}}}}
}}
```

## Contact

For questions or issues, please contact [your-email@example.com]
"""
    
    return model_card

def upload_model_to_hub(model_path: Path,
                       repo_name: str,
                       training_info: Dict,
                       performance_metrics: Dict,
                       private: bool = True) -> Optional[str]:
    """Upload a trained model to Hugging Face Hub."""
    
    print(f"\n📤 Uploading model to Hugging Face Hub...")
    print(f"Repository: {repo_name}")
    
    try:
        # Load model to get config
        model = HFNodeDetector.from_pretrained(str(model_path))
        
        # Create model card
        model_card = create_model_card(model.config, training_info, performance_metrics)
        
        # Save model card
        readme_path = model_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(model_card)
        
        # Create additional metadata
        metadata = {
            "framework": "pytorch",
            "task": "image-classification",
            "dataset_info": training_info,
            "performance": performance_metrics,
            "model_config": {
                "architecture": model.config.architecture_type,
                "num_labels": model.config.num_labels,
                "learning_rate": model.config.learning_rate
            }
        }
        
        metadata_path = model_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Upload to Hub
        api = HfApi()
        
        # Create repository
        repo_url = api.create_repo(
            repo_id=repo_name,
            private=private,
            repo_type="model",
            exist_ok=True
        )
        
        # Upload files
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            repo_type="model",
            commit_message="Upload MRI lymph node classification model"
        )
        
        model_url = f"https://huggingface.co/{repo_name}"
        print(f"✓ Model uploaded successfully!")
        print(f"  URL: {model_url}")
        return model_url
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

def download_and_test_model(repo_name: str, test_data_path: Optional[Path] = None):
    """Download and test a model from Hugging Face Hub."""
    
    print(f"\n📥 Downloading model from Hugging Face Hub...")
    print(f"Repository: {repo_name}")
    
    try:
        # Download model
        model = HFNodeDetector.from_pretrained(repo_name)
        print(f"✓ Model downloaded successfully!")
        print(f"  Architecture: {model.config.architecture_type}")
        print(f"  Num labels: {model.config.num_labels}")
        
        # Test inference
        print(f"\n🔍 Testing inference...")
        
        # Create test data if not provided
        if test_data_path is None or not test_data_path.exists():
            print("  Creating dummy test data...")
            mri_volume = torch.randn(1, 1, 64, 64, 64)
            segmentation = torch.randint(0, 2, (1, 1, 64, 64, 64)).float()
        else:
            # Load real test data (implementation depends on your data format)
            print(f"  Loading test data from {test_data_path}")
            # TODO: Implement actual data loading
            mri_volume = torch.randn(1, 1, 64, 64, 64)
            segmentation = torch.randint(0, 2, (1, 1, 64, 64, 64)).float()
        
        # Run inference
        model.eval()
        with torch.no_grad():
            outputs = model(image=mri_volume, segmentation=segmentation)
            probabilities = torch.sigmoid(outputs['logits'])
            prediction = (probabilities > 0.5).long()
        
        print(f"  ✓ Inference successful!")
        print(f"    Prediction: {prediction.item()}")
        print(f"    Confidence: {probabilities.item():.3f}")
        
        return model
        
    except Exception as e:
        print(f"❌ Download/test failed: {e}")
        return None

def create_model_comparison_app(model_repos: List[str], output_path: Path):
    """Create a simple web app to compare models from different repositories."""
    
    app_code = f'''
import streamlit as st
import torch
from mri_node_validator.models.hf_detector import HFNodeDetector

st.title("🏥 MRI Lymph Node Classification Model Comparison")

# Model selection
model_options = {model_repos}
selected_model = st.selectbox("Select Model", model_options)

# Load model
@st.cache_resource
def load_model(repo_name):
    return HFNodeDetector.from_pretrained(repo_name)

if selected_model:
    with st.spinner("Loading model..."):
        model = load_model(selected_model)
    
    st.success(f"✓ Model loaded: {{selected_model}}")
    
    # Display model info
    st.subheader("Model Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Architecture", model.config.architecture_type)
        st.metric("Number of Labels", model.config.num_labels)
    
    with col2:
        st.metric("Learning Rate", f"{{model.config.learning_rate:.1e}}")
    
    # File upload for inference
    st.subheader("Upload MRI Data for Inference")
    
    uploaded_file = st.file_uploader(
        "Choose a NIfTI file", 
        type=['nii', 'nii.gz'],
        help="Upload a 3D MRI volume in NIfTI format"
    )
    
    if uploaded_file is not None:
        # Process uploaded file
        st.info("Processing uploaded file...")
        
        # Create dummy data for demo (replace with actual NIfTI processing)
        mri_volume = torch.randn(1, 1, 64, 64, 64)
        segmentation = torch.randint(0, 2, (1, 1, 64, 64, 64)).float()
        
        # Run inference
        model.eval()
        with torch.no_grad():
            outputs = model(image=mri_volume, segmentation=segmentation)
            probabilities = torch.sigmoid(outputs['logits'])
            prediction = (probabilities > 0.5).long()
        
        # Display results
        st.subheader("Inference Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Prediction", 
                "Merged Nodes" if prediction.item() == 1 else "No Merged Nodes"
            )
        
        with col2:
            confidence = probabilities.item() if prediction.item() == 1 else (1 - probabilities.item())
            st.metric("Confidence", f"{{confidence:.1%}}")
        
        # Visualization placeholder
        st.subheader("Volume Visualization")
        st.info("Visualization feature coming soon...")

# Run with: streamlit run app.py
'''
    
    app_path = output_path / "streamlit_app.py"
    with open(app_path, 'w') as f:
        f.write(app_code)
    
    print(f"✓ Streamlit app created: {app_path}")
    print(f"  Run with: streamlit run {app_path}")

def main():
    print("🤗 Hugging Face Hub Integration for MRI Models")
    print("=" * 50)
    
    # Setup authentication
    is_authenticated = setup_huggingface_auth()
    
    # Example 1: Upload a trained model (requires authentication)
    if is_authenticated:
        print(f"\n📤 Example 1: Upload Model to Hub")
        
        # Check if we have a trained model
        model_path = Path("./trained_model")
        if model_path.exists():
            # Sample training info and metrics (would come from actual training)
            training_info = {
                "num_samples": 100,
                "input_shape": "[64, 64, 64]",
                "target_spacing": "[1.0, 1.0, 1.0]",
                "batch_size": 4,
                "num_epochs": 10,
                "augmentation": "rotation, flipping, noise, elastic deformation"
            }
            
            performance_metrics = {
                "accuracy": 0.89,
                "precision": 0.87,
                "recall": 0.91,
                "f1": 0.89,
                "auc": 0.92
            }
            
            # Get username for repo name
            try:
                api = HfApi()
                username = api.whoami()['name']
                repo_name = f"{username}/mri-lymph-node-classifier"
                
                # Upload model
                model_url = upload_model_to_hub(
                    model_path=model_path,
                    repo_name=repo_name,
                    training_info=training_info,
                    performance_metrics=performance_metrics,
                    private=True  # Set to False to make public
                )
                
                if model_url:
                    print(f"\n✅ Upload completed! Your model is available at:")
                    print(f"   {model_url}")
                    
            except Exception as e:
                print(f"❌ Upload example failed: {e}")
        else:
            print("  ⚠️  No trained model found. Run training example first.")
    
    # Example 2: Download and test a model
    print(f"\n📥 Example 2: Download and Test Model")
    
    # Example public models (replace with actual model repos when available)
    example_repos = [
        # "username/mri-lymph-node-classifier",  # Would be real repo
        # "medical-ai/mri-node-detector",        # Example public repo
    ]
    
    if example_repos:
        for repo in example_repos:
            print(f"\nTesting model: {repo}")
            model = download_and_test_model(repo)
    else:
        print("  📝 No example repositories configured.")
        print("     Replace example_repos with actual model repositories.")
    
    # Example 3: Create a simple comparison app
    print(f"\n🌐 Example 3: Create Model Comparison App")
    
    output_dir = Path("./hub_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create example app
    create_model_comparison_app(
        model_repos=["username/model-v1", "username/model-v2"],
        output_path=output_dir
    )
    
    # Example 4: Best practices for model sharing
    print(f"\n📋 Example 4: Best Practices for Model Sharing")
    
    best_practices = '''
🌟 BEST PRACTICES FOR SHARING MRI MODELS ON HF HUB:

1. 📝 Documentation:
   - Comprehensive model cards with performance metrics
   - Clear usage examples and code snippets
   - Detailed training procedure and hyperparameters
   - Limitations and ethical considerations

2. 🔒 Data Privacy:
   - Ensure all training data is properly anonymized
   - No patient-identifying information in models or metadata
   - Follow HIPAA and GDPR guidelines

3. 🏷️ Metadata & Tags:
   - Use descriptive tags: medical-imaging, mri, classification
   - Include dataset information (size, demographics)
   - Specify scanner types and protocols used

4. 🔄 Versioning:
   - Use semantic versioning (v1.0.0, v1.1.0, etc.)
   - Document changes between versions
   - Maintain backward compatibility when possible

5. 🧪 Evaluation:
   - Include comprehensive evaluation metrics
   - Test on diverse datasets when possible
   - Provide confidence intervals for metrics

6. 🛡️ Safety:
   - Clear warnings about clinical use limitations
   - Recommend professional medical review
   - Include bias and fairness assessments

7. 🤝 Community:
   - Respond to issues and questions promptly
   - Accept contributions and improvements
   - Provide citation information
'''
    
    print(best_practices)
    
    # Example usage code
    usage_example = '''
# 🚀 QUICK START EXAMPLE:

from mri_node_validator.models.hf_detector import HFNodeDetector
import torch

# 1. Load model from Hub
model = HFNodeDetector.from_pretrained("username/mri-lymph-node-classifier")

# 2. Prepare your MRI data
mri_volume = torch.randn(1, 1, 64, 64, 64)      # Your NIfTI data
segmentation = torch.randn(1, 1, 64, 64, 64)    # Segmentation mask

# 3. Run inference
model.eval()
with torch.no_grad():
    outputs = model(image=mri_volume, segmentation=segmentation)
    probabilities = torch.sigmoid(outputs['logits'])
    prediction = (probabilities > 0.5).long()

# 4. Interpret results
class_names = ["No Merged Nodes", "Merged Nodes"]
result = class_names[prediction.item()]
confidence = probabilities.item() if prediction.item() == 1 else (1 - probabilities.item())

print(f"Prediction: {result}")
print(f"Confidence: {confidence:.1%}")
'''
    
    # Save usage example
    example_path = output_dir / "usage_example.py"
    with open(example_path, 'w') as f:
        f.write(usage_example)
    
    print(f"\n💾 Usage example saved to: {example_path}")
    
    print(f"\n🎉 Hugging Face Hub integration examples completed!")
    print(f"\n📁 Files created in: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Train a model using the training examples")
    print(f"  2. Upload to Hugging Face Hub with proper authentication")
    print(f"  3. Share with the medical AI community!")

if __name__ == "__main__":
    main()