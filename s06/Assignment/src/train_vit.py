"""
Training script for CatDog ViT Classifier with CometML logging.
"""
import os
import json
from pathlib import Path

import hydra
import lightning as L
import torch
import rootutils
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import CometLogger, CSVLogger
from loguru import logger

# Setup root directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.catdog_datamodule import CatDogDataModule
from src.vit_model import ViTClassifier


def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_training_curves(csv_path, output_dir):
    """Plot training and validation curves."""
    import pandas as pd
    
    # Read metrics from CSV logger
    metrics_file = Path(csv_path) / "metrics.csv"
    if not metrics_file.exists():
        logger.warning(f"Metrics file not found: {metrics_file}")
        return
    
    df = pd.read_csv(metrics_file)
    
    # Plot accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_acc = df[df['train/acc_epoch'].notna()][['epoch', 'train/acc_epoch']].dropna()
    val_acc = df[df['val/acc'].notna()][['epoch', 'val/acc']].dropna()
    
    if not train_acc.empty:
        ax.plot(train_acc['epoch'], train_acc['train/acc_epoch'], 'b-', label='Train Accuracy', marker='o')
    if not val_acc.empty:
        ax.plot(val_acc['epoch'], val_acc['val/acc'], 'r-', label='Val Accuracy', marker='s')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_plot.png", dpi=150)
    plt.close()
    
    # Plot loss
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_loss = df[df['train/loss_epoch'].notna()][['epoch', 'train/loss_epoch']].dropna()
    val_loss = df[df['val/loss'].notna()][['epoch', 'val/loss']].dropna()
    
    if not train_loss.empty:
        ax.plot(train_loss['epoch'], train_loss['train/loss_epoch'], 'b-', label='Train Loss', marker='o')
    if not val_loss.empty:
        ax.plot(val_loss['epoch'], val_loss['val/loss'], 'r-', label='Val Loss', marker='s')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_plot.png", dpi=150)
    plt.close()
    
    logger.info(f"Saved training curves to {output_dir}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_vit")
def main(cfg: DictConfig):
    """Main training function."""
    logger.info("="*50)
    logger.info("Starting CatDog ViT Training")
    logger.info("="*50)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set seed for reproducibility
    L.seed_everything(cfg.seed, workers=True)
    
    # Create output directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/plots", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/predictions", exist_ok=True)
    
    # Initialize DataModule
    datamodule = CatDogDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
    )
    datamodule.prepare_data()
    datamodule.setup()
    
    # Initialize Model
    model = ViTClassifier(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.model.learning_rate,
        pretrained=cfg.model.pretrained,
        class_names=list(datamodule.class_to_idx.keys()),
    )
    
    # Setup loggers
    loggers = []
    
    # CSV Logger (always)
    csv_logger = CSVLogger(save_dir=cfg.output_dir, name="csv_logs")
    loggers.append(csv_logger)
    
    # CometML Logger (if API key available)
    comet_api_key = os.environ.get("COMET_API_KEY")
    if comet_api_key:
        comet_logger = CometLogger(
            api_key=comet_api_key,
            project_name=cfg.get("comet_project", "catdog-vit"),
            experiment_name=cfg.get("experiment_name", "vit-training"),
        )
        loggers.append(comet_logger)
        logger.info("CometML logging enabled")
    else:
        logger.warning("COMET_API_KEY not found, skipping CometML logging")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.output_dir}/checkpoints",
        filename="catdog-vit-{epoch:02d}-{val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    
    early_stopping = EarlyStopping(
        monitor="val/acc",
        patience=cfg.trainer.patience,
        mode="max",
    )
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback, early_stopping, RichProgressBar()],
        logger=loggers,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=True,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model, datamodule)
    
    # Test
    logger.info("Running test evaluation...")
    trainer.test(model, datamodule, ckpt_path="best")
    
    # Save confusion matrices
    train_cm = model.get_train_confusion_matrix().cpu().numpy()
    test_cm = model.get_test_confusion_matrix().cpu().numpy()
    
    class_names = list(datamodule.class_to_idx.keys())
    
    plot_confusion_matrix(
        train_cm, 
        class_names, 
        "Training Confusion Matrix",
        f"{cfg.output_dir}/plots/train_confusion_matrix.png"
    )
    
    plot_confusion_matrix(
        test_cm, 
        class_names, 
        "Test Confusion Matrix",
        f"{cfg.output_dir}/plots/test_confusion_matrix.png"
    )
    
    # Plot training curves
    plot_training_curves(
        csv_logger.log_dir,
        f"{cfg.output_dir}/plots"
    )
    
    # Save best model path
    best_model_path = checkpoint_callback.best_model_path
    with open(f"{cfg.output_dir}/best_model_path.txt", 'w') as f:
        f.write(best_model_path)
    
    # Save final metrics
    metrics = {
        "best_val_acc": float(checkpoint_callback.best_model_score),
        "test_acc": float(trainer.callback_metrics.get("test/acc", 0)),
        "test_loss": float(trainer.callback_metrics.get("test/loss", 0)),
    }
    
    with open(f"{cfg.output_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("="*50)
    logger.info("Training Complete!")
    logger.info(f"Best validation accuracy: {metrics['best_val_acc']:.4f}")
    logger.info(f"Test accuracy: {metrics['test_acc']:.4f}")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info("="*50)
    
    return metrics


if __name__ == "__main__":
    main()
