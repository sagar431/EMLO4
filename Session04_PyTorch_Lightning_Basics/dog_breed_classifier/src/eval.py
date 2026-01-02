import json
import os
from pathlib import Path

import lightning as L
from src.train import DogBreedClassifier
from src.datamodule import DogBreedDataModule
from src.utils.logging_config import setup_logging
from loguru import logger
from rich.console import Console
from rich.table import Table

def main():
    setup_logging(log_file="logs/evaluation.log")
    logger.info("Starting evaluation process")

    # Load best model path
    with open('best_model_path.txt', 'r') as f:
        checkpoint_path = f.read().strip()
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    # Data module setup
    data_dir = os.getenv('DATA_DIR', 'data/dog-breed-dataset')
    batch_size = int(os.getenv('BATCH_SIZE', '32'))
    num_workers = int(os.getenv('NUM_WORKERS', '4'))
    
    datamodule = DogBreedDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    datamodule.setup()

    # Load model
    model = DogBreedClassifier.load_from_checkpoint(checkpoint_path)
    logger.info("Model loaded successfully")

    # Trainer setup
    trainer = L.Trainer(
        accelerator='auto',
        devices=1,
        callbacks=[L.callbacks.RichProgressBar()],
    )

    # Run evaluation
    logger.info("Starting evaluation")
    results = trainer.validate(model, datamodule=datamodule)
    
    # Log results
    metrics = {
        'val_loss': float(results[0]['val_loss']),
        'val_accuracy': float(results[0]['val_acc'])
    }
    
    # Save metrics to file
    os.makedirs('predictions', exist_ok=True)
    metrics_file = 'predictions/eval_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Validation Results:")
    logger.info(f"Loss: {metrics['val_loss']:.4f}")
    logger.info(f"Accuracy: {metrics['val_accuracy']:.4f}")
    logger.info(f"Metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()
