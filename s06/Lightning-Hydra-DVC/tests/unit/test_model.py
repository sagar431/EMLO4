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


def test_timm_classifier_training_step(cfg_model: DictConfig):
    """Test that training_step returns a valid loss tensor."""
    model = TimmClassifier(
        base_model=cfg_model.base_model,
        num_classes=cfg_model.num_classes,
        pretrained=cfg_model.pretrained,
        lr=cfg_model.lr,
    )
    
    # Create a dummy batch (images, labels)
    batch = (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 0, 1]))
    
    loss = model.training_step(batch, batch_idx=0)
    
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Loss should be a scalar
    assert loss.requires_grad  # Loss should be differentiable


def test_timm_classifier_validation_step(cfg_model: DictConfig):
    """Test that validation_step runs without error."""
    model = TimmClassifier(
        base_model=cfg_model.base_model,
        num_classes=cfg_model.num_classes,
        pretrained=cfg_model.pretrained,
        lr=cfg_model.lr,
    )
    
    # Create a dummy batch (images, labels)
    batch = (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 0, 1]))
    
    # validation_step should not raise any error
    result = model.validation_step(batch, batch_idx=0)
    # validation_step returns None (logs metrics internally)
    assert result is None


def test_timm_classifier_configure_optimizers(cfg_model: DictConfig):
    """Test that configure_optimizers returns proper optimizer config."""
    model = TimmClassifier(
        base_model=cfg_model.base_model,
        num_classes=cfg_model.num_classes,
        pretrained=cfg_model.pretrained,
        lr=cfg_model.lr,
    )
    
    opt_config = model.configure_optimizers()
    
    assert "optimizer" in opt_config
    assert "lr_scheduler" in opt_config
    assert opt_config["lr_scheduler"]["monitor"] == "val/loss" 