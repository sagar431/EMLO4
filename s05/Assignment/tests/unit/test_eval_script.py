"""Tests for eval.py script functions."""
import pytest
import os
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.eval import setup_logging


class TestEvalSetupLogging:
    """Test cases for eval.py setup_logging function."""
    
    def test_setup_logging_without_file(self):
        """Test setup_logging without log file."""
        setup_logging()
        # Should not raise an error
        
    def test_setup_logging_with_file(self, tmp_path):
        """Test setup_logging with log file."""
        log_file = str(tmp_path / "logs" / "test.log")
        setup_logging(log_file=log_file)
        
        # Directory should be created
        assert (tmp_path / "logs").exists()


class TestEvalHelpers:
    """Test helper functions from eval.py."""
    
    def test_import_eval_module(self):
        """Test that eval module can be imported."""
        from src import eval as eval_module
        assert eval_module is not None
    
    def test_eval_has_main_function(self):
        """Test that eval has main function."""
        from src import eval as eval_module
        assert hasattr(eval_module, 'main')
        assert callable(eval_module.main)
    
    def test_eval_has_setup_logging(self):
        """Test that eval has setup_logging function."""
        from src import eval as eval_module
        assert hasattr(eval_module, 'setup_logging')


class TestInferSetupLogging:
    """Test cases for infer.py setup_logging function."""
    
    def test_infer_import(self):
        """Test that infer module can be imported."""
        from src import infer as infer_module
        assert infer_module is not None
    
    def test_infer_has_main(self):
        """Test that infer has main function."""
        from src import infer as infer_module
        assert hasattr(infer_module, 'main')
    
    def test_infer_has_process_image(self):
        """Test that infer has process_image function."""
        from src import infer as infer_module
        assert hasattr(infer_module, 'process_image')
    
    def test_infer_has_get_image_files(self):
        """Test that infer has get_image_files function."""
        from src import infer as infer_module
        assert hasattr(infer_module, 'get_image_files')
    
    def test_infer_has_display_predictions(self):
        """Test that infer has display_predictions function."""
        from src import infer as infer_module
        assert hasattr(infer_module, 'display_predictions')


class TestInferHelpers:
    """Test infer.py helper functions."""
    
    def test_get_image_files_empty_folder(self, tmp_path):
        """Test get_image_files with empty folder."""
        from src.infer import get_image_files
        
        files = get_image_files(str(tmp_path), ['.jpg', '.png'], 10)
        assert len(files) == 0
    
    def test_get_image_files_with_images(self, tmp_path):
        """Test get_image_files with some images."""
        from src.infer import get_image_files
        
        # Create test images
        (tmp_path / "test1.jpg").touch()
        (tmp_path / "test2.png").touch()
        (tmp_path / "test3.txt").touch()  # Should be ignored
        
        files = get_image_files(str(tmp_path), ['.jpg', '.png'], 10)
        assert len(files) == 2
    
    def test_get_image_files_max_limit(self, tmp_path):
        """Test get_image_files respects max_images limit."""
        from src.infer import get_image_files
        
        # Create many test images
        for i in range(10):
            (tmp_path / f"test{i}.jpg").touch()
        
        files = get_image_files(str(tmp_path), ['.jpg'], 5)
        assert len(files) == 5
    
    def test_process_image_returns_tuple(self, tmp_path):
        """Test process_image returns correct tuple."""
        from src.infer import process_image
        from torchvision import transforms
        from PIL import Image
        
        # Create a test image
        img_path = tmp_path / "test.jpg"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        tensor, original = process_image(img_path, transform)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
        assert isinstance(original, Image.Image)


class TestInferLogging:
    """Test infer.py logging setup."""
    
    def test_infer_setup_logging_without_file(self):
        """Test infer setup_logging without file."""
        from src.infer import setup_logging
        setup_logging()
        # Should not raise
    
    def test_infer_setup_logging_with_file(self, tmp_path):
        """Test infer setup_logging with file."""
        from src.infer import setup_logging
        log_file = str(tmp_path / "logs" / "infer.log")
        setup_logging(log_file=log_file)
        assert (tmp_path / "logs").exists()
