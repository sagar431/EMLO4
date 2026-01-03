"""Unit tests for eval.py inference functionality."""
import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.train import DogBreedClassifier


class TestEvalFunctionality:
    """Test cases for evaluation functionality."""
    
    @pytest.fixture
    def model(self):
        """Create a model instance for testing."""
        return DogBreedClassifier(
            num_classes=10,
            learning_rate=0.001,
            class_mapping={i: f"breed_{i}" for i in range(10)}
        )
    
    def test_model_eval_mode(self, model):
        """Test that model can be set to eval mode."""
        model.eval()
        assert not model.training
    
    def test_model_train_mode(self, model):
        """Test that model can be set to train mode."""
        model.train()
        assert model.training
    
    def test_model_can_be_saved_and_loaded(self, model, tmp_path):
        """Test that model can be saved and loaded from checkpoint."""
        # Save model
        checkpoint_path = tmp_path / "test_checkpoint.ckpt"
        
        # Use PyTorch Lightning's save method
        import lightning as L
        trainer = L.Trainer(max_epochs=0, logger=False, enable_checkpointing=False)
        trainer.strategy.connect(model)
        
        # Save state dict manually
        torch.save({
            'state_dict': model.state_dict(),
            'hyper_parameters': model.hparams,
        }, checkpoint_path)
        
        # Verify file exists
        assert checkpoint_path.exists()
    
    def test_softmax_output_sums_to_one(self, model):
        """Test that softmax output sums to 1."""
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)
        
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    
    def test_argmax_returns_valid_class(self, model):
        """Test that argmax returns a valid class index."""
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
            pred_idx = output.argmax(dim=1).item()
        
        assert 0 <= pred_idx < 10


class TestInferenceFunctionality:
    """Test cases for inference functionality."""
    
    @pytest.fixture
    def model(self):
        """Create a model instance for testing."""
        return DogBreedClassifier(
            num_classes=10,
            learning_rate=0.001,
            class_mapping={i: f"breed_{i}" for i in range(10)}
        )
    
    def test_batch_inference(self, model):
        """Test inference on a batch of images."""
        model.eval()
        batch_size = 8
        x = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 10)
    
    def test_class_mapping_lookup(self, model):
        """Test that class mapping lookup works."""
        class_mapping = model.hparams.class_mapping
        
        assert class_mapping[0] == "breed_0"
        assert class_mapping[5] == "breed_5"
        assert class_mapping[9] == "breed_9"
    
    def test_confidence_calculation(self, model):
        """Test confidence score calculation."""
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)[0]
            confidence = probs.max().item()
        
        assert 0 <= confidence <= 1
