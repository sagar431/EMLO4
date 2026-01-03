"""Tests for datamodule.py."""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.datamodule import DogBreedDataModule


class TestDogBreedDataModule:
    """Test cases for DogBreedDataModule."""
    
    def test_datamodule_initialization(self):
        """Test that datamodule initializes correctly."""
        dm = DogBreedDataModule(
            data_dir="data/test",
            batch_size=32,
            num_workers=4
        )
        
        assert dm.data_dir == "data/test"
        assert dm.batch_size == 32
        assert dm.num_workers == 4
    
    def test_datamodule_default_values(self):
        """Test default values for datamodule."""
        dm = DogBreedDataModule()
        
        assert dm.batch_size == 32  # Default batch size
        assert dm.num_workers == 4  # Default num workers
    
    def test_datamodule_custom_batch_size(self):
        """Test custom batch size."""
        dm = DogBreedDataModule(batch_size=64)
        assert dm.batch_size == 64
    
    def test_datamodule_custom_num_workers(self):
        """Test custom num workers."""
        dm = DogBreedDataModule(num_workers=8)
        assert dm.num_workers == 8
    
    def test_datamodule_has_prepare_data(self):
        """Test that datamodule has prepare_data method."""
        dm = DogBreedDataModule()
        assert hasattr(dm, 'prepare_data')
        assert callable(dm.prepare_data)
    
    def test_datamodule_has_setup(self):
        """Test that datamodule has setup method."""
        dm = DogBreedDataModule()
        assert hasattr(dm, 'setup')
        assert callable(dm.setup)
    
    def test_datamodule_has_train_dataloader(self):
        """Test that datamodule has train_dataloader method."""
        dm = DogBreedDataModule()
        assert hasattr(dm, 'train_dataloader')
        assert callable(dm.train_dataloader)
    
    def test_datamodule_has_val_dataloader(self):
        """Test that datamodule has val_dataloader method."""
        dm = DogBreedDataModule()
        assert hasattr(dm, 'val_dataloader')
        assert callable(dm.val_dataloader)
