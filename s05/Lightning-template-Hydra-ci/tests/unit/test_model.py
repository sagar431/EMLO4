import pytest
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from src.models.timm_classifier import TimmClassifier


@pytest.fixture
def cfg_model() -> DictConfig:
    """Load the default model config."""
    with initialize(config_path="../../configs/model", version_base="1.3"):
        cfg = compose(config_name="timm_classify.yaml")
        # Convert to a simple dict to avoid Hydra interpolation issues
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    return cfg


def test_timm_classifier_forward(cfg_model: DictConfig):
    """Test TimmClassifier forward pass."""
    model = TimmClassifier(
        base_model=cfg_model.base_model,
        num_classes=cfg_model.num_classes,
        pretrained=cfg_model.pretrained,
        lr=cfg_model.lr,
    )
    # Create a dummy input tensor (batch_size, channels, height, width)
    # Assuming input size expected by resnet18, adjust if using a different model
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    assert output is not None
    # Check output shape: (batch_size, num_classes)
    assert output.shape == (2, cfg_model.num_classes) 