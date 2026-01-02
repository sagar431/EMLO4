import lightning as L
from pathlib import Path
from typing import Union
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datasets import load_dataset
import logging

class CatDogImageDataModule(L.LightningDataModule):
    def __init__(self, dl_path: Union[str, Path] = "data", num_workers: int = 0, batch_size: int = 8):
        super().__init__()
        self._dl_path = Path(dl_path)
        self._num_workers = num_workers
        self._batch_size = batch_size
        
        # Ensure the data directory exists
        os.makedirs(self._dl_path, exist_ok=True)

    def prepare_data(self):
        """Download images from Hugging Face and prepare datasets."""
        try:
            train_path = self.data_path / "train"
            val_path = self.data_path / "validation"
            
            if not train_path.exists() or not val_path.exists():
                logging.info("Downloading cats_vs_dogs dataset from Hugging Face...")
                dataset = load_dataset("microsoft/cats_vs_dogs", split="train")
                
                # Create directories
                for split_path in [train_path, val_path]:
                    (split_path / "cats").mkdir(parents=True, exist_ok=True)
                    (split_path / "dogs").mkdir(parents=True, exist_ok=True)
                
                # Split into train (80%) and validation (20%)
                dataset = dataset.shuffle(seed=42)
                split_idx = int(len(dataset) * 0.8)
                
                for idx, sample in enumerate(dataset):
                    image = sample["image"]
                    label = sample["labels"]  # 0 = cat, 1 = dog
                    
                    # Determine split and class folder
                    split_folder = train_path if idx < split_idx else val_path
                    class_folder = "cats" if label == 0 else "dogs"
                    
                    # Save image
                    image_path = split_folder / class_folder / f"{idx}.jpg"
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image.save(image_path)
                
                logging.info(f"Dataset saved to {self.data_path}")
        except Exception as e:
            logging.error(f"Failed to download or extract dataset: {str(e)}")
            raise

    @property
    def data_path(self):
        return self._dl_path / "cats_and_dogs_filtered"

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((160, 160)),  # Reduced image size
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def valid_transform(self):
        return transforms.Compose([
            transforms.Resize((160, 160)),  # Reduced image size
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def create_dataset(self, root, transform):
        if not root.exists():
            raise RuntimeError(f"Dataset directory {root} does not exist!")
        return ImageFolder(root=root, transform=transform)

    def __dataloader(self, train: bool):
        """Train/validation/test loaders."""
        try:
            if train:
                dataset = self.create_dataset(self.data_path.joinpath("train"), self.train_transform)
            else:
                dataset = self.create_dataset(self.data_path.joinpath("validation"), self.valid_transform)
            
            return DataLoader(
                dataset=dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                shuffle=train,
                pin_memory=True  # Enabled for GPU - faster data transfer
            )
        except Exception as e:
            logging.error(f"Failed to create dataloader: {str(e)}")
            raise

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)
        
    def test_dataloader(self):
        return self.__dataloader(train=False) 