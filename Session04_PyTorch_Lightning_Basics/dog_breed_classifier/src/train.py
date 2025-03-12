import os
from pathlib import Path
from typing import Dict, Optional

import lightning as L
import torch
import torch.nn.functional as F
from src.datamodule import DogBreedDataModule
from loguru import logger
from rich.console import Console
from rich.table import Table
from torchvision import models

class DogBreedClassifier(L.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3, class_mapping=None):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pretrained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, num_classes)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.class_mapping = class_mapping

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def main():
    setup_logging()
    logger.info("Starting training process")

    # Data module setup
    data_dir = os.getenv('DATA_DIR', 'data/dog-breed-dataset')
    batch_size = int(os.getenv('BATCH_SIZE', '32'))
    num_workers = int(os.getenv('NUM_WORKERS', '4'))
    
    datamodule = DogBreedDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    datamodule.setup()

    # Get class mapping
    class_mapping = datamodule.train_dataset.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    # Model setup
    num_classes = len(class_mapping)
    model = DogBreedClassifier(
        num_classes=num_classes,
        class_mapping=idx_to_class  # Pass the mapping to the model
    )
    
    logger.info(f"Created model with {num_classes} classes")
    logger.info(f"Class mapping: {class_mapping}")

    # Callbacks
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        dirpath='checkpoints',
        filename='dog-breed-{epoch:02d}-{val_acc:.2f}',
        monitor='val_acc',
        mode='max',
        save_top_k=1
    )

    # Trainer setup
    trainer = L.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=1,
        callbacks=[
            L.callbacks.RichProgressBar(),
            L.callbacks.RichModelSummary(max_depth=2),
            checkpoint_callback
        ],
        logger=True
    )

    # Train
    logger.info("Starting model training")
    trainer.fit(model, datamodule=datamodule)
    
    # Save best model path
    best_model_path = checkpoint_callback.best_model_path
    with open('best_model_path.txt', 'w') as f:
        f.write(best_model_path)
    logger.info(f"Training completed. Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main()
