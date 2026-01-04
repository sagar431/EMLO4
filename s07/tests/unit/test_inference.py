"""Tests for the inference script."""
import pytest
import torch
from pathlib import Path
from PIL import Image
import numpy as np

from src.models.timm_classifier import TimmClassifier


@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    model = TimmClassifier(
        base_model="resnet18",
        num_classes=2,
        pretrained=False,  # Don't download weights for testing speed
        lr=1e-3,
    )
    model.eval()
    return model


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample test image."""
    img_path = tmp_path / "test_image.jpg"
    # Create a simple random RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="RGB")
    img.save(img_path)
    return img_path


def test_model_inference_output_shape(dummy_model, sample_image):
    """Test that model produces correct output shape during inference."""
    from torchvision import transforms
    
    # Load and transform image
    img = Image.open(sample_image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = dummy_model(img_tensor)
    
    assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"


def test_model_inference_returns_probabilities(dummy_model, sample_image):
    """Test that model output can be converted to valid probabilities."""
    from torchvision import transforms
    import torch.nn.functional as F
    
    # Load and transform image
    img = Image.open(sample_image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = dummy_model(img_tensor)
        probs = F.softmax(output, dim=1)
    
    # Check probabilities sum to 1
    prob_sum = probs.sum().item()
    assert 0.99 <= prob_sum <= 1.01, f"Probabilities should sum to 1, got {prob_sum}"
    
    # Check all probabilities are between 0 and 1
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities should be in [0, 1]"


def test_model_prediction_is_valid_class(dummy_model, sample_image):
    """Test that model prediction is a valid class index."""
    from torchvision import transforms
    import torch.nn.functional as F
    
    # Load and transform image
    img = Image.open(sample_image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = dummy_model(img_tensor)
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
    
    # Check predicted class is valid (0 for cat, 1 for dog)
    assert predicted_class in [0, 1], f"Predicted class should be 0 or 1, got {predicted_class}"


def test_inference_on_batch(dummy_model):
    """Test that model can handle batch inference."""
    batch_size = 4
    batch = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        output = dummy_model(batch)
    
    assert output.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {output.shape}"
