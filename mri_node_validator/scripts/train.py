#!/usr/bin/env python3
"""
Training script using Hugging Face Trainer with custom MRI data loading pipeline.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from data.dataset import MRIDataModule
from data.transforms import get_preprocessing_pipeline
from data.collator import MRIDataCollator
from models.detector import NodeDetector
from models.hf_detector import HFNodeDetector, MRIConfig


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""
    
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    model_type: str = field(
        default="3d_resnet_attention",
        metadata={"help": "Model architecture type"}
    )
    num_classes: int = field(
        default=2,
        metadata={"help": "Number of classes for classification"}
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"help": "Learning rate for training"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    
    data_dir: str = field(
        metadata={"help": "Directory containing NIfTI files"}
    )
    csv_path: str = field(
        metadata={"help": "Path to CSV file containing labels"}
    )
    key_pattern: str = field(
        default=r"(.+)\.nii(?:\.gz)?$",
        metadata={"help": "Regex pattern to extract key from filename"}
    )
    label_column: str = field(
        default="label",
        metadata={"help": "Column name for labels in CSV"}
    )
    file_key_column: str = field(
        default="file_key",
        metadata={"help": "Column name for file keys in CSV"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of training samples to use"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of evaluation samples to use"}
    )
    train_split: float = field(
        default=0.7,
        metadata={"help": "Proportion of data for training"}
    )
    val_split: float = field(
        default=0.15,
        metadata={"help": "Proportion of data for validation"}
    )
    test_split: float = field(
        default=0.15,
        metadata={"help": "Proportion of data for testing"}
    )


class MRITrainer(Trainer):
    """Custom Trainer class for MRI classification with torchio data loading."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training/evaluation loss.
        
        Args:
            model: The model to train
            inputs: The inputs and targets of the model
            return_outputs: Whether to return the model outputs along with the loss
            
        Returns:
            The loss tensor, and optionally the outputs
        """
        # Remove non-model inputs
        model_inputs = {k: v for k, v in inputs.items() 
                       if k not in ['file_key', 'file_path']}
        
        outputs = model(**model_inputs)
        loss = outputs.get('loss')
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred: EvalPrediction object containing predictions and labels
        
    Returns:
        Dictionary of computed metrics
    """
    predictions, labels = eval_pred
    
    # Handle binary classification (single output with sigmoid)
    if predictions.ndim == 1 or (predictions.ndim > 1 and predictions.shape[1] == 1):
        # Binary: apply sigmoid threshold
        if predictions.ndim > 1:
            probs = torch.sigmoid(torch.tensor(predictions)).numpy().flatten()
        else:
            probs = torch.sigmoid(torch.tensor(predictions)).numpy()
        predictions = (probs > 0.5).astype(int)
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.0
    else:
        # Multi-class: use argmax
        predictions = np.argmax(predictions, axis=1)
        try:
            probs = torch.softmax(torch.tensor(eval_pred.predictions), dim=1)[:, 1]
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.0
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_data_module(data_args: DataArguments, config: Dict[str, Any]) -> MRIDataModule:
    """Setup the data module with transforms."""
    
    # Get preprocessing pipeline from config
    preprocessing_config = config.get('preprocessing', {})
    train_transforms, val_transforms, test_transforms = get_preprocessing_pipeline(preprocessing_config)
    
    # Create data module
    data_module = MRIDataModule(
        data_dir=data_args.data_dir,
        csv_path=data_args.csv_path,
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 4),
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=test_transforms,
        train_split=data_args.train_split,
        val_split=data_args.val_split,
        test_split=data_args.test_split,
        random_seed=config.get('seed', 42),
        key_pattern=data_args.key_pattern,
        label_column=data_args.label_column,
        file_key_column=data_args.file_key_column
    )
    
    return data_module


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train MRI classifier using Hugging Face Trainer")
    
    # Add arguments
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing NIfTI files")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file with labels")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for model and logs")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Create arguments objects
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        **config.get('model', {})
    )
    
    data_args = DataArguments(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        **config.get('data', {})
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        **config.get('training', {})
    )
    
    # Setup data module
    data_module = setup_data_module(data_args, config)
    train_dataset, eval_dataset, test_dataset = data_module.setup()
    
    # Print dataset info
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize HF-compatible model
    config = MRIConfig(
        learning_rate=model_args.learning_rate,
        model_type=model_args.model_type,
        num_labels=model_args.num_classes
    )
    
    if model_args.model_name_or_path:
        # Load from HF checkpoint
        model = HFNodeDetector.from_pretrained(model_args.model_name_or_path)
    else:
        # Create new HF model
        model = HFNodeDetector(config)
    
    # Setup data collator
    data_collator = MRIDataCollator()
    
    # Setup trainer
    trainer = MRITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    # Training
    if args.do_train:
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        elif os.path.isdir(training_args.output_dir):
            checkpoint = get_last_checkpoint(training_args.output_dir)
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    if args.do_eval:
        print("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        
        metrics["eval_samples"] = len(eval_dataset)
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Prediction
    if args.do_predict:
        print("*** Predict ***")
        predictions = trainer.predict(test_dataset=test_dataset)
        
        # Save predictions
        output_predict_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        with open(output_predict_file, "w") as writer:
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions.predictions):
                if len(item) > 1:
                    # Multi-class: get argmax
                    prediction = np.argmax(item)
                else:
                    # Binary: threshold at 0.5
                    prediction = 1 if item[0] > 0.5 else 0
                writer.write(f"{index}\t{prediction}\n")
        
        print(f"Predictions saved to {output_predict_file}")


if __name__ == "__main__":
    main()