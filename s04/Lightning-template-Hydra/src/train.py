import os
import torch
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# Enable Tensor Core optimization for better GPU performance
# Options: 'highest' (most precise), 'high', 'medium' (fastest)
torch.set_float32_matmul_precision('medium')

from datamodules.catdog_datamodule import CatDogImageDataModule
from models.catdog_classifier import CatDogClassifier
from utils.rich_utils import setup_logger

def main():
    # Setup logger
    logger = setup_logger()
    logger.info("Starting training...")

    # Set seed for reproducibility
    L.seed_everything(42, workers=True)

    # Initialize callbacks
    callbacks = [
        RichProgressBar(),
        RichModelSummary(max_depth=2),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="logs/catdog_classification",
            filename="epoch={epoch}-step={step}",
            save_top_k=3,
            mode="min",
        ),
    ]

    # Configure for GPU environment
    logger.info("Configuring for GPU environment...")
    
    # Initialize DataModule with GPU-optimized configuration
    logger.info("Initializing DataModule...")
    data_module = CatDogImageDataModule(
        batch_size=16,  # Optimized for GTX 1650 (4GB VRAM)
        num_workers=2,  # Balanced for your system
        dl_path="data"  # Explicitly set data path
    )

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Initialize Model
    logger.info("Initializing Model...")
    model = CatDogClassifier(lr=1e-4)  # Lower LR for stability

    # Initialize Trainer with GPU configuration
    logger.info("Setting up Trainer...")
    trainer = Trainer(
        max_epochs=5,
        callbacks=callbacks,
        accelerator="gpu",  # Use GPU
        devices=1,          # Use 1 GPU
        precision=32,       # Use 32-bit for stability (GTX 1650 has limited FP16 support)
        gradient_clip_val=1.0,  # Prevent exploding gradients
        logger=TensorBoardLogger(save_dir="logs", name="catdog_classification"),
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    try:
        # Train and test the model
        logger.info("Starting training...")
        trainer.fit(model, data_module)
        
        logger.info("Starting testing...")
        trainer.test(model, data_module)
        
        logger.info("Training completed!")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
