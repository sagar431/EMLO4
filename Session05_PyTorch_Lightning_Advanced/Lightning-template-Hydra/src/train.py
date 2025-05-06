import os
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

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

    # Use minimal resources for Gitpod environment
    logger.info("Configuring for CPU-only environment...")
    
    # Initialize DataModule with minimal worker configuration
    logger.info("Initializing DataModule...")
    data_module = CatDogImageDataModule(
        batch_size=8,  # Reduced batch size
        num_workers=0,  # No multiprocessing
        dl_path="data"  # Explicitly set data path
    )

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Initialize Model
    logger.info("Initializing Model...")
    model = CatDogClassifier(lr=1e-3)

    # Initialize Trainer with CPU configuration
    logger.info("Setting up Trainer...")
    trainer = Trainer(
        max_epochs=5,
        callbacks=callbacks,
        accelerator="cpu",  # Force CPU
        devices=1,
        logger=TensorBoardLogger(save_dir="logs", name="catdog_classification"),
        deterministic=True,
        detect_anomaly=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        # Limit batches for testing
        limit_train_batches=0.2,  # Use 20% of training data
        limit_val_batches=0.2,    # Use 20% of validation data
        limit_test_batches=0.2,   # Use 20% of test data
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
