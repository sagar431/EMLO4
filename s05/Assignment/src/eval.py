"""
Evaluation script with Hydra configuration.
Tests the model given a checkpoint and saves metrics.
"""
import json
import os
from pathlib import Path

import hydra
import lightning as L
import rootutils
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from rich.console import Console
from rich.table import Table

# Setup root directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.train import DogBreedClassifier
from src.datamodule import DogBreedDataModule


def setup_logging(log_file: str = None):
    """Configure loguru logging."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(log_file, rotation="10 MB")


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> dict:
    """
    Main evaluation function.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Dictionary containing evaluation metrics
    """
    setup_logging(log_file=f"{cfg.output_dir}/evaluation.log")
    
    logger.info("Starting evaluation process")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Resolve checkpoint path
    checkpoint_path = cfg.checkpoint_path
    if not os.path.exists(checkpoint_path):
        # Try to find best_model_path.txt
        best_path_file = Path(cfg.paths.root_dir) / "best_model_path.txt"
        if best_path_file.exists():
            with open(best_path_file, 'r') as f:
                checkpoint_path = f.read().strip()
            logger.info(f"Using checkpoint from best_model_path.txt: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Instantiate datamodule
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    logger.info("DataModule setup complete")
    
    # Load model from checkpoint
    model = DogBreedClassifier.load_from_checkpoint(checkpoint_path)
    logger.info("Model loaded successfully")
    
    # Setup trainer
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[L.callbacks.RichProgressBar()],
        logger=False,
    )
    
    # Run evaluation
    logger.info("Starting evaluation on validation set")
    results = trainer.validate(model, datamodule=datamodule)
    
    # Extract metrics
    metrics = {
        'val_loss': float(results[0]['val_loss']),
        'val_accuracy': float(results[0]['val_acc'])
    }
    
    # Also run test if available
    if hasattr(datamodule, 'test_dataloader') and datamodule.test_dataloader() is not None:
        logger.info("Running evaluation on test set")
        test_results = trainer.test(model, datamodule=datamodule)
        if test_results:
            metrics['test_loss'] = float(test_results[0].get('test_loss', 0))
            metrics['test_accuracy'] = float(test_results[0].get('test_acc', 0))
    
    # Create output directory and save metrics
    os.makedirs(cfg.output_dir, exist_ok=True)
    metrics_file = cfg.metrics_file
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Display results in a nice table
    console = Console()
    table = Table(title="📊 Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    for key, value in metrics.items():
        if 'accuracy' in key.lower() or 'acc' in key.lower():
            table.add_row(key, f"{value:.4f} ({value*100:.2f}%)")
        else:
            table.add_row(key, f"{value:.4f}")
    
    console.print(table)
    
    logger.info(f"Metrics saved to: {metrics_file}")
    logger.info("Evaluation complete!")
    
    return metrics


if __name__ == "__main__":
    main()
