"""Fashion MNIST DataModule for PyTorch Lightning."""

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from pathlib import Path


class FashionMNISTDataModule(L.LightningDataModule):
    """Fashion MNIST DataModule for image classification."""
    
    # Class names for Fashion MNIST
    CLASSES = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.1,
        image_size: int = 224,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.image_size = image_size
        
        # Transforms - Fashion MNIST is 28x28 grayscale, need to resize and convert to 3 channels
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """Download data if needed."""
        # FashionMNIST will download if not present
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        """Setup train, val, and test datasets."""
        if stage == "fit" or stage is None:
            full_train = FashionMNIST(
                self.data_dir, train=True, transform=self.train_transform
            )
            
            # Split into train and validation
            val_size = int(len(full_train) * self.val_split)
            train_size = len(full_train) - val_size
            
            self.train_dataset, val_temp = random_split(
                full_train, [train_size, val_size]
            )
            
            # Apply val transform to validation set
            self.val_dataset = FashionMNIST(
                self.data_dir, train=True, transform=self.val_transform
            )
            # Use same indices as val_temp
            self.val_dataset = [self.val_dataset[i] for i in val_temp.indices]
        
        if stage == "test" or stage is None:
            self.test_dataset = FashionMNIST(
                self.data_dir, train=False, transform=self.val_transform
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
