"""Tests for configuration validation."""
import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf


@pytest.fixture
def main_cfg():
    """Load the main training config."""
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="train.yaml")
    return cfg


def test_all_configs_compose(main_cfg):
    """Verify all config files can be composed without errors."""
    assert main_cfg is not None
    
    # Check required top-level keys exist
    assert "data" in main_cfg
    assert "model" in main_cfg
    assert "trainer" in main_cfg
    assert "paths" in main_cfg
    assert "callbacks" in main_cfg


def test_data_config_has_target(main_cfg):
    """Verify data config has _target_ for instantiation."""
    assert "_target_" in main_cfg.data
    assert "CatDogImageDataModule" in main_cfg.data._target_


def test_model_config_has_target(main_cfg):
    """Verify model config has _target_ for instantiation."""
    assert "_target_" in main_cfg.model
    assert "TimmClassifier" in main_cfg.model._target_


def test_trainer_config_has_target(main_cfg):
    """Verify trainer config has _target_ for instantiation."""
    assert "_target_" in main_cfg.trainer
    assert "Trainer" in main_cfg.trainer._target_


def test_model_config_valid_hyperparameters(main_cfg):
    """Verify model hyperparameters are within valid ranges."""
    assert main_cfg.model.num_classes >= 2, "num_classes should be at least 2"
    assert 0 < main_cfg.model.lr < 1, "Learning rate should be between 0 and 1"
    assert main_cfg.model.base_model is not None, "base_model should be specified"


def test_trainer_config_valid_epochs(main_cfg):
    """Verify trainer epochs configuration is valid."""
    assert main_cfg.trainer.max_epochs >= 1, "max_epochs should be at least 1"
    assert main_cfg.trainer.min_epochs >= 1, "min_epochs should be at least 1"
    assert main_cfg.trainer.min_epochs <= main_cfg.trainer.max_epochs, \
        "min_epochs should be <= max_epochs"


def test_data_config_valid_splits(main_cfg):
    """Verify data splits sum to approximately 1.0."""
    splits = main_cfg.data.splits
    total = sum(splits)
    assert 0.99 <= total <= 1.01, f"Splits should sum to 1.0, got {total}"
    assert all(s > 0 for s in splits), "All splits should be positive"


def test_config_override_from_command_line():
    """Test that command-line overrides work correctly."""
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["model.lr=0.01", "trainer.max_epochs=5"]
        )
        assert cfg.model.lr == 0.01
        assert cfg.trainer.max_epochs == 5
