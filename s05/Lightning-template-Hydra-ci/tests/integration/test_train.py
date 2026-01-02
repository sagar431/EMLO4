import pytest
import os
from pathlib import Path
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

# Adjust the import path according to your project structure
from src.train import main as train_main


@pytest.fixture
def hydra_cfg() -> DictConfig:
    """Load the default hydra config for training."""
    with initialize(config_path="../../configs", version_base="1.3"):
        # Compose the config but don't initialize HydraConfig
        cfg = compose(config_name="train.yaml")
    return cfg


def test_train_fast_dev_run(hydra_cfg: DictConfig, tmp_path):
    """Test the main training script with fast_dev_run=True."""
    # Create a processed config with resolved values and overrides
    processed_config = process_config_for_testing(hydra_cfg, tmp_path, fast_dev_run=True)
    
    try:
        train_main(processed_config)
    except Exception as e:
        pytest.fail(f"Training script failed with fast_dev_run=True: {e}")


def test_train_and_test_run(hydra_cfg: DictConfig, tmp_path):
    """Test the main training script for a short train and test run."""
    # Create a processed config with resolved values and overrides
    processed_config = process_config_for_testing(
        hydra_cfg, 
        tmp_path,
        trainer_overrides={
            "max_epochs": 1,
            "limit_train_batches": 2,
            "limit_val_batches": 2,
            "limit_test_batches": 2
        }
    )
    
    try:
        train_main(processed_config)
    except Exception as e:
        pytest.fail(f"Training script failed during train/test run: {e}")


def process_config_for_testing(
    cfg: DictConfig, 
    tmp_path: Path, 
    fast_dev_run: bool = False,
    trainer_overrides: dict = None
) -> DictConfig:
    """
    Process the Hydra config for testing by:
    1. Creating a clean copy
    2. Resolving all interpolations that might depend on HydraConfig
    3. Setting up test-specific overrides
    
    Args:
        cfg: The original config from hydra.compose()
        tmp_path: Pytest fixture for temporary directory
        fast_dev_run: Whether to enable fast_dev_run
        trainer_overrides: Additional trainer config overrides
        
    Returns:
        Processed config ready for testing
    """
    # Create a deep copy to avoid modifying the original
    config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    
    # Create test output directories
    test_output_dir = os.path.abspath(tmp_path / "hydra_testing")
    log_dir = os.path.abspath(tmp_path / "lightning_logs")
    os.makedirs(test_output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Resolve paths and hydra interpolations
    with open_dict(config):
        # Set paths
        if "paths" in config:
            config.paths.output_dir = test_output_dir
            config.paths.log_dir = log_dir
            config.paths.work_dir = os.getcwd()
        
        # Disable all callbacks to avoid path interpolation issues
        config.callbacks = None
        
        # Disable loggers for testing
        config.logger = None
        
        # Configure trainer
        if fast_dev_run:
            config.trainer.fast_dev_run = True
        
        if trainer_overrides:
            for k, v in trainer_overrides.items():
                OmegaConf.update(config, f"trainer.{k}", v)
    
    return config

# You might want to add more specific tests, e.g.:
# - Test if checkpoints are created (if not using fast_dev_run)
# - Test specific callback behavior
# - Test loading from a checkpoint 