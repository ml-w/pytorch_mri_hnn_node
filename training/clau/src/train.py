import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
from tqdm import tqdm
import torchmetrics
import os

# Import from our custom modules
from utils import seed_everything, CONFIG
from dataset import NiftiDataset, get_transforms

def get_model(pretrained: bool = True):
    """
    Loads a pretrained ResNet-50 and adapts it for binary classification.
    """
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    else:
        weights = None
        
    model = models.resnet50(weights=weights)
    
    # Adapt the final layer for binary classification (1 output logit)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    
    return model

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, freeze_backbone: bool):
    """
    Trains the model for one epoch.
    """
    model.train()
    
    # Set requires_grad for parameters based on freeze strategy
    for name, param in model.named_parameters():
        if freeze_backbone and name.startswith("fc"):
            param.requires_grad = True
        elif freeze_backbone:
            param.requires_grad = False
        else:
            param.requires_grad = True

    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1) # Ensure labels are [B, 1]
        
        optimizer.zero_grad()
        
        # Mixed precision context
        with torch.cuda.amp.autocast(enabled=CONFIG["training"]["use_amp"]):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, metrics_collection):
    """
    Validates the model on the given dataloader.
    """
    model.eval()
    total_loss = 0.0
    metrics_collection.reset()
    
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Update metrics
            preds = torch.sigmoid(outputs)
            metrics_collection.update(preds, labels.int())

    avg_loss = total_loss / len(dataloader)
    computed_metrics = metrics_collection.compute()
    return avg_loss, computed_metrics


def main():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup & Reproducibility
    seed_everything(CONFIG["training"]["seed"])
    device = torch.device(CONFIG["training"]["device"])
    os.makedirs(CONFIG["output"]["dir"], exist_ok=True)
    
    # 2. Load Data & Patient-Level Splitting
    df = pd.read_csv(CONFIG["data"]["labels_csv"])
    
    # Use StratifiedGroupKFold for robust, patient-aware splitting
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CONFIG["training"]["seed"])
    splits = list(cv.split(X=df, y=df['label'], groups=df['patient_id']))
    
    train_idx, val_idx = [], []
    for fold_idx in CONFIG["data"]["train_folds"]:
        train_idx.extend(splits[fold_idx][0])
    for fold_idx in CONFIG["data"]["val_folds"]:
        val_idx.extend(splits[fold_idx][1])
        
    # 3. Create Datasets & DataLoaders
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    full_dataset = NiftiDataset(df, root_dir=CONFIG["data"]["root_dir"], transform=None) # No transform yet
    
    # Apply transforms within dataset subsets
    train_dataset = Subset(full_dataset, train_idx)
    train_dataset.dataset.transform = train_transform
    
    val_dataset = Subset(full_dataset, val_idx)
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=True,
        num_workers=CONFIG["training"]["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False,
        num_workers=CONFIG["training"]["num_workers"],
        pin_memory=True
    )
    
    print(f"Data prepared: {len(train_dataset)} train samples, {len(val_dataset)} validation samples.")

    # 4. Model, Loss, Optimizer
    model = get_model(pretrained=CONFIG["model"]["pretrained"]).to(device)
    
    # Calculate pos_weight for imbalanced datasets
    train_labels = df.iloc[train_idx]['label']
    pos_weight = torch.tensor(train_labels.value_counts()[0] / train_labels.value_counts()[1], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["training"]["learning_rate"], weight_decay=CONFIG["training"]["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1, verbose=True)
    
    # 5. Metrics & Training Loop
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["training"]["use_amp"])
    
    # Use torchmetrics for efficient metric calculation
    metrics = torchmetrics.MetricCollection({
        'accuracy': torchmetrics.Accuracy(task='binary'),
        'auroc': torchmetrics.AUROC(task='binary'),
    }).to(device)

    best_auroc = 0.0
    epochs_no_improve = 0
    
    for epoch in range(CONFIG["training"]["num_epochs"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['training']['num_epochs']} ---")
        
        # Freeze/unfreeze strategy
        freeze_backbone = epoch < CONFIG["model"]["freeze_epochs"]
        if freeze_backbone:
            print("Mode: Training FC layer only.")
        else:
            if epoch == CONFIG["model"]["freeze_epochs"]:
                 print("Mode: Unfreezing backbone and training all layers.")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, freeze_backbone)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, metrics)
        
        auroc = val_metrics['auroc'].item()
        accuracy = val_metrics['accuracy'].item()
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy:.4f} | Val AUROC: {auroc:.4f}")
        
        scheduler.step(auroc) # Step scheduler on validation AUROC
        
        # Save best model checkpoint
        if auroc > best_auroc:
            best_auroc = auroc
            epochs_no_improve = 0
            checkpoint_path = os.path.join(CONFIG["output"]["dir"], "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path} (AUROC: {best_auroc:.4f})")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
            break
            
    print("\nTraining finished.")
    print(f"Best validation AUROC: {best_auroc:.4f}")
    # The best model is saved at "project/outputs/best_model.pth"
    # You can now load it and evaluate on the test set.

if __name__ == '__main__':
    main()