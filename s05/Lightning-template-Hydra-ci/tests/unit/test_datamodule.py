import pytest
import os
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict

from src.datamodules.catdog_datamodule import CatDogImageDataModule


@pytest.fixture
def cfg_datamodule() -> DictConfig:
    """Load the default datamodule config."""
    with initialize(config_path="../../configs", version_base="1.3"):
        # Load the main config to get paths and then get data config
        main_cfg = compose(config_name="train.yaml")
        
        # Then get the data config directly
        data_cfg = main_cfg.data
        
        # Ensure it's modifiable and resolve interpolations
        data_cfg = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True))
        
    return data_cfg


def test_catdog_datamodule(cfg_datamodule: DictConfig):
    """Test CatDogImageDataModule setup and dataset lengths."""
    # Use a test-specific data directory
    datamodule = CatDogImageDataModule(
        data_dir="data/",  # Use a simple relative path for testing
        batch_size=cfg_datamodule.batch_size,
        num_workers=cfg_datamodule.num_workers,
        splits=cfg_datamodule.splits,
    )
    datamodule.prepare_data()
    datamodule.setup()

    assert datamodule.train_dataset
    assert datamodule.val_dataset
    assert datamodule.test_dataset

    # Calculate expected lengths based on actual dataset size (not hardcoded!)
    # The Hugging Face microsoft/cats_vs_dogs dataset has ~18,727 images
    total_images = len(datamodule._dataset)
    train_len = int(cfg_datamodule.splits[0] * total_images)
    val_len = int(cfg_datamodule.splits[1] * total_images)
    test_len = total_images - train_len - val_len

    assert len(datamodule.train_dataset) == train_len, f"Train dataset length mismatch. Expected {train_len}, got {len(datamodule.train_dataset)}"
    assert len(datamodule.val_dataset) == val_len, f"Validation dataset length mismatch. Expected {val_len}, got {len(datamodule.val_dataset)}"
    assert len(datamodule.test_dataset) == test_len, f"Test dataset length mismatch. Expected {test_len}, got {len(datamodule.test_dataset)}"

    # Check dataloader creation
    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader() 