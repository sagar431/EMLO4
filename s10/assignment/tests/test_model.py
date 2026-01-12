"""
Tests for the Cat-Dog classifier model.
"""

import pytest
import torch
import os
from pathlib import Path


class TestModelTracing:
    """Tests for model tracing functionality."""
    
    def test_torch_available(self):
        """Test that PyTorch is available."""
        assert torch.__version__ is not None
        print(f"PyTorch version: {torch.__version__}")
    
    def test_cuda_availability(self):
        """Test CUDA availability (informational)."""
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    def test_model_creation(self):
        """Test that model can be created."""
        import timm
        
        model = timm.create_model('resnet18', pretrained=False, num_classes=2)
        assert model is not None
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 2)
        print(f"Model output shape: {output.shape}")
    
    def test_model_tracing(self):
        """Test that model can be traced."""
        import timm
        
        model = timm.create_model('resnet18', pretrained=False, num_classes=2)
        model.eval()
        
        # Trace the model
        example_input = torch.randn(1, 3, 224, 224)
        traced_model = torch.jit.trace(model, example_input)
        
        assert traced_model is not None
        
        # Verify outputs match
        with torch.no_grad():
            original_output = model(example_input)
            traced_output = traced_model(example_input)
        
        assert torch.allclose(original_output, traced_output, rtol=1e-4, atol=1e-4)
        print("✅ Traced model outputs match original!")
    
    def test_traced_model_save_load(self, tmp_path):
        """Test that traced model can be saved and loaded."""
        import timm
        
        model = timm.create_model('resnet18', pretrained=False, num_classes=2)
        model.eval()
        
        # Trace and save
        example_input = torch.randn(1, 3, 224, 224)
        traced_model = torch.jit.trace(model, example_input)
        
        save_path = tmp_path / "test_model.pt"
        torch.jit.save(traced_model, save_path)
        
        assert save_path.exists()
        print(f"Model saved to: {save_path}")
        
        # Load and verify
        loaded_model = torch.jit.load(save_path)
        
        with torch.no_grad():
            loaded_output = loaded_model(example_input)
            original_output = traced_model(example_input)
        
        assert torch.allclose(original_output, loaded_output, rtol=1e-4, atol=1e-4)
        print("✅ Loaded model outputs match!")


class TestGradioApp:
    """Tests for Gradio app functionality."""
    
    def test_gradio_import(self):
        """Test that Gradio can be imported."""
        import gradio as gr
        assert gr.__version__ is not None
        print(f"Gradio version: {gr.__version__}")
    
    def test_transforms(self):
        """Test that transforms work correctly."""
        import torchvision.transforms as transforms
        from PIL import Image
        import numpy as np
        
        # Create dummy image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)
        print(f"Transform output shape: {tensor.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
