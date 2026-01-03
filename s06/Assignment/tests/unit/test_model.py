"""Unit tests for the DogBreedClassifier model."""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.train import DogBreedClassifier


class TestDogBreedClassifier:
    """Test cases for DogBreedClassifier."""
    
    @pytest.fixture
    def model(self):
        """Create a model instance for testing."""
        return DogBreedClassifier(
            num_classes=10,
            learning_rate=0.001,
            class_mapping={i: f"breed_{i}" for i in range(10)}
        )
    
    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert model is not None
        assert model.hparams.num_classes == 10
        assert model.hparams.learning_rate == 0.001
        assert len(model.hparams.class_mapping) == 10
    
    def test_forward_pass(self, model):
        """Test forward pass with random input."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 10)
    
    def test_training_step(self, model):
        """Test training step."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        y = torch.randint(0, 10, (batch_size,))
        
        loss = model.training_step((x, y), 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Loss should be positive
    
    def test_validation_step(self, model):
        """Test validation step."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        y = torch.randint(0, 10, (batch_size,))
        
        result = model.validation_step((x, y), 0)
        
        assert isinstance(result, dict)
        assert 'val_loss' in result
        assert 'val_acc' in result
    
    def test_configure_optimizers(self, model):
        """Test optimizer configuration."""
        optimizer = model.configure_optimizers()
        
        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Adam)
    
    def test_model_output_probabilities(self, model):
        """Test that softmax produces valid probabilities."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            output = model(x)
            probabilities = torch.softmax(output, dim=1)
        
        # Check probabilities sum to 1
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        # Check all probabilities are positive
        assert (probabilities >= 0).all()
    
    def test_class_mapping(self, model):
        """Test class mapping functionality."""
        assert model.class_mapping is not None
        assert 0 in model.class_mapping
        assert model.class_mapping[0] == "breed_0"
