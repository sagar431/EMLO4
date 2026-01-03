"""
ViT Classifier Model for CatDog classification.
"""
import lightning as L
import torch
import torch.nn.functional as F
import timm
from torchmetrics import Accuracy, ConfusionMatrix
from loguru import logger


class ViTClassifier(L.LightningModule):
    """Vision Transformer Classifier using timm."""
    
    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        pretrained: bool = True,
        class_names: list = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.class_names = class_names or ["Cat", "Dog"]
        
        # Create ViT model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        logger.info(f"Created {model_name} with pretrained={pretrained}")
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Confusion matrices for plotting
        self.train_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.test_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        # Store predictions for confusion matrix
        self.train_preds = []
        self.train_targets = []
        self.test_preds = []
        self.test_targets = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        
        # Update metrics
        self.train_acc(preds, y)
        
        # Store for confusion matrix
        self.train_preds.append(preds.cpu())
        self.train_targets.append(y.cpu())
        
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        
        self.val_acc(preds, y)
        
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        
        self.test_acc(preds, y)
        
        # Store for confusion matrix
        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())
        
        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc)
        
        return loss
    
    def on_train_epoch_end(self):
        # Compute confusion matrix for train set
        if self.train_preds:
            all_preds = torch.cat(self.train_preds)
            all_targets = torch.cat(self.train_targets)
            self.train_confmat.update(all_preds, all_targets)
            self.train_preds = []
            self.train_targets = []
    
    def on_test_epoch_end(self):
        # Compute confusion matrix for test set
        if self.test_preds:
            all_preds = torch.cat(self.test_preds)
            all_targets = torch.cat(self.test_targets)
            self.test_confmat.update(all_preds, all_targets)
    
    def get_train_confusion_matrix(self):
        return self.train_confmat.compute()
    
    def get_test_confusion_matrix(self):
        return self.test_confmat.compute()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=5,
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
