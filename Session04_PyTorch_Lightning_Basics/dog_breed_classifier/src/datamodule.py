import os
from typing import Optional
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from loguru import logger

class DogBreedDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = os.path.join(data_dir, 'dataset')  # Updated path to include 'dataset' subdirectory
        self.transform = transform
        self.image_files = []
        self.labels = []
        
        # Get all breed directories
        breed_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        self.class_to_idx = {breed: idx for idx, breed in enumerate(sorted(breed_dirs))}
        
        # Collect all image paths and labels
        for breed in breed_dirs:
            breed_path = os.path.join(self.data_dir, breed)
            for img_name in os.listdir(breed_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_files.append(os.path.join(breed_path, img_name))
                    self.labels.append(self.class_to_idx[breed])
        
        logger.info(f"Found {len(self.image_files)} images across {len(breed_dirs)} breeds")
        logger.info(f"Class mapping: {self.class_to_idx}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class DogBreedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = 'data/dog-breed-dataset',
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: float = 0.2
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

        self.transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            full_dataset = DogBreedDataset(self.data_dir, transform=self.transform_train)
            train_size = int((1 - self.train_val_split) * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            # Save class mapping for inference
            self.class_to_idx = full_dataset.class_to_idx
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            self.val_dataset.dataset.transform = self.transform_val
            
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            logger.info(f"Validation dataset size: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
