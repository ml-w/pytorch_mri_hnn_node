import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Tuple
from torchmetrics import Accuracy, Precision, Recall, F1Score

class NodeDetector(pl.LightningModule):
    """PyTorch Lightning module for detecting merged lymph nodes."""
    
    def __init__(self, 
                 learning_rate: float = 1e-3,
                 model_type: str = "3d_resnet_attention"):
        super().__init__()
        self.save_hyperparameters()
        
        # Base feature extractor
        self.backbone = torch.hub.load(
            'facebookresearch/3d-resnet', 
            'resnet18',
            pretrained=True)
            
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Metrics
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.train_precision = Precision(task='binary')
        self.val_precision = Precision(task='binary')
        self.train_recall = Recall(task='binary')
        self.val_recall = Recall(task='binary')
        self.train_f1 = F1Score(task='binary')
        self.val_f1 = F1Score(task='binary')
        
    def forward(self, 
               image: torch.Tensor,
               segmentation: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference.
        
        Args:
            image: Input MRI volume (B, 1, D, H, W)
            segmentation: Input segmentation mask (B, 1, D, H, W)
            
        Returns:
            Probability scores (B, 1)
        """
        x = torch.cat([image, segmentation], dim=1)
        features = self.backbone(x)
        return self.classifier(features)
        
    def training_step(self, batch, batch_idx):
        """Training step with metrics logging."""
        image, seg, target = batch
        logits = self(image, seg)
        loss = F.binary_cross_entropy(logits, target.float())
        
        # Update metrics
        self.train_acc(logits, target)
        self.train_precision(logits, target)
        self.train_recall(logits, target)
        self.train_f1(logits, target)
        
        self.log_dict({
            'train_loss': loss,
            'train_acc': self.train_acc,
            'train_precision': self.train_precision,
            'train_recall': self.train_recall,
            'train_f1': self.train_f1
        }, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Validation step with metrics logging."""
        image, seg, target = batch
        logits = self(image, seg)
        loss = F.binary_cross_entropy(logits, target.float())
        
        # Update metrics
        self.val_acc(logits, target)
        self.val_precision(logits, target)
        self.val_recall(logits, target)
        self.val_f1(logits, target)
        
        self.log_dict({
            'val_loss': loss,
            'val_acc': self.val_acc,
            'val_precision': self.val_precision,
            'val_recall': self.val_recall,
            'val_f1': self.val_f1
        }, prog_bar=True)
        
        return loss
        
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), 
                                   lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=3,
            verbose=True)
            
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }