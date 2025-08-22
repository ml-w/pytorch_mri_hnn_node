import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
from tqdm import tqdm
import torchmetrics
import os
import torchio as tio

from utils import seed_everything, CONFIG
from dataset import get_transforms, permute_and_squeeze

def get_model(pretrained: bool = True):
    """
    Loads a pretrained ResNet-50 and adapts it for binary classification.
    """
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    else:
        weights = None
        
    model = models.resnet50(weights=weights)
    
    # Adapt the final layer for binary classification
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    
    return model

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    """
    Trains the model for one epoch.
    """
    model.train()

    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        images = batch['mri']['data']  # Shape: (B, 1, H, W, 3)
        images = images.squeeze(1).permute(0, 3, 1, 2).to(device)  # Shape: (B, 3, H, W)
        labels = batch['label'].to(device).unsqueeze(1).float()  # Shape: (B, 1)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=CONFIG["training"]["use_amp"]):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
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
        for batch in progress_bar:
            images = batch['mri']['data']  # Shape: (B, 1, H, W, 3)
            images = images.squeeze(1).permute(0, 3, 1, 2).to(device)  # Shape: (B, 3, H, W)
            labels = batch['label'].to(device).unsqueeze(1).float()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            metrics_collection.update(preds, labels.long())

    avg_loss = total_loss / len(dataloader)
    computed_metrics = metrics_collection.compute()
    return avg_loss, computed_metrics

def main():
    """
    Main function to run the training pipeline with cross-validation.
    """
    # 1. Setup & Reproducibility
    seed_everything(CONFIG["training"]["seed"])
    device = torch.device(CONFIG["training"]["device"])
    os.makedirs(CONFIG["output"]["dir"], exist_ok=True)
    
    # 2. Load Data
    df = pd.read_csv(CONFIG["data"]["labels_csv"])
    
    # 3. Define Cross-Validation Strategy
    n_splits = CONFIG["training"]["n_splits"]
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=CONFIG["training"]["seed"])

    # 4. Start Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(cv.split(X=df, y=df['individual'], groups=df['patient_id'])):
        print(f"\n--- Starting Fold {fold+1}/{n_splits} ---")
                
        # 5. Create Datasets & DataLoaders for the current fold
        
        # patch_level_transform = tio.Lambda(
        #     # permute_and_squeeze,
        #     types_to_apply=['mri']
        # )
        
        # train_volume_transforms = get_transforms(train=True)
        # train_transform_list = list(train_volume_transforms.transforms) + [patch_level_transform]
        # train_transform = tio.Compose(train_transform_list)
        
        # val_volume_transforms = get_transforms(train=False)
        # val_transform_list = list(val_volume_transforms.transforms) + [patch_level_transform]
        # val_transform = tio.Compose(val_transform_list)

        train_transform = get_transforms(train=True)
        val_transform = get_transforms(train=False)

        train_subjects = [
            tio.Subject(
                mri=tio.ScalarImage(os.path.join(CONFIG["data"]["root_dir"], row.filename)), 
                label=row.individual
            ) 
            for row in df.iloc[train_idx].itertuples()
        ]
        train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG["training"]["batch_size"],
            shuffle=True,
            num_workers=CONFIG["training"]["num_workers"],
            pin_memory=True
        )
        
        val_subjects = [
            tio.Subject(
                mri=tio.ScalarImage(os.path.join(CONFIG["data"]["root_dir"], row.filename)), 
                label=row.individual
            ) 
            for row in df.iloc[val_idx].itertuples()
        ]
        val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=CONFIG["training"]["batch_size"], 
            shuffle=False,
            num_workers=CONFIG["training"]["num_workers"],
            pin_memory=True
        )
        
        print(f"Fold {fold+1} Data: {len(train_dataset)} train samples, {len(val_dataset)} validation samples.")

        # 6. Re-initialize Model, Loss, Optimizer for each fold
        model = get_model(pretrained=CONFIG["model"]["pretrained"]).to(device)
        
        # Calculate pos_weight for the current training data
        train_labels = df.iloc[train_idx]['individual']
        counts = train_labels.value_counts()
        pos_weight_val = counts.get(0, 1) / counts.get(1, 1)
        pos_weight = torch.tensor(pos_weight_val, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["training"]["learning_rate"], weight_decay=CONFIG["training"]["weight_decay"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=CONFIG["training"]["scheduler_mode"], 
            patience=CONFIG["training"]["scheduler_patience"], 
            factor=CONFIG["training"]["scheduler_factor"], 
            verbose=True
        )       

        # 7. Metrics & Training Loop
        scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["training"]["use_amp"])
        metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task='binary'),
            'auroc': torchmetrics.AUROC(task='binary'),
            'sensitivity': torchmetrics.Recall(task='binary'),
            'specificity': torchmetrics.Specificity(task='binary'),
            'f1': torchmetrics.F1Score(task='binary'),
        }).to(device)

        best_auroc = 0.0
        epochs_no_improve = 0

        print("Mode: Starting with FC layer training only.")
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
        
        for epoch in range(CONFIG["training"]["num_epochs"]):
            print(f"\n--- Fold {fold+1}, Epoch {epoch+1}/{CONFIG['training']['num_epochs']} ---")
            
            if epoch == CONFIG["model"]["freeze_epochs"]:
                print("Mode: Unfreezing backbone and training all layers.")
                for param in model.parameters():
                    param.requires_grad = True
            
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_loss, val_metrics = validate(model, val_loader, criterion, device, metrics)
            
            auroc = val_metrics['auroc'].item()
            accuracy = val_metrics['accuracy'].item()
            sensitivity = val_metrics['sensitivity'].item()
            specificity = val_metrics['specificity'].item()
            f1 = val_metrics['f1'].item()
            
            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val AUROC: {auroc:.4f} | Val Acc: {accuracy:.4f} | "
                f"Val Sensitivity: {sensitivity:.4f} | Val Specificity: {specificity:.4f} | Val F1: {f1:.4f}"
            )            
            scheduler.step(auroc)
            
            if auroc > best_auroc:
                best_auroc = auroc
                epochs_no_improve = 0
                checkpoint_path = os.path.join(CONFIG["output"]["dir"], f"best_model_fold_{fold+1}.pth")
                checkpoint = {
                    'fold': fold + 1,
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_auroc': best_auroc,
                    'config': CONFIG,
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path} (AUROC: {best_auroc:.4f})")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= CONFIG["training"]["patience"]:
                print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
                break
                
        print(f"\nTraining for Fold {fold+1} finished.")
        print(f"Best validation AUROC for Fold {fold+1}: {best_auroc:.4f}")

        del model, optimizer, scheduler, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()