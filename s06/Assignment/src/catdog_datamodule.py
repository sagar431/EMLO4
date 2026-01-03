"""
CatDog DataModule for PyTorch Lightning.
"""
import os
from pathlib import Path
from typing import Optional

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from loguru import logger
from PIL import Image


def is_valid_image(path):
    """Check if image is valid (not corrupted)."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False


class CatDogDataModule(L.LightningDataModule):
    """DataModule for Cats vs Dogs dataset."""
    
    def __init__(
        self,
        data_dir: str = "data/PetImages",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        train_val_split: float = 0.8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_val_split = train_val_split
        
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_to_idx = {"Cat": 0, "Dog": 1}
        self.idx_to_class = {0: "Cat", 1: "Dog"}
    
    def prepare_data(self):
        """Clean corrupted images from dataset."""
        data_path = Path(self.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Remove corrupted images
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        if not is_valid_image(img_path):
                            logger.warning(f"Removing corrupted image: {img_path}")
                            img_path.unlink()
    
    def setup(self, stage: Optional[str] = None):
        """Setup train, val, and test datasets."""
        # Create full dataset
        full_dataset = ImageFolder(
            self.data_dir,
            transform=self.train_transform,
            is_valid_file=is_valid_image
        )
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(total_size * self.train_val_split * 0.9)  # 72% train
        val_size = int(total_size * self.train_val_split * 0.1)    # 8% val
        test_size = total_size - train_size - val_size             # 20% test
        
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # Apply val transform to val and test sets
        self.val_dataset.dataset.transform = self.val_transform
        self.test_dataset.dataset.transform = self.val_transform
        
        self.class_to_idx = full_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        logger.info(f"Train size: {len(self.train_dataset)}")
        logger.info(f"Val size: {len(self.val_dataset)}")
        logger.info(f"Test size: {len(self.test_dataset)}")
        logger.info(f"Classes: {self.class_to_idx}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
