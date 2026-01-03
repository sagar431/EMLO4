"""
Inference script for CatDog ViT Classifier.
Generates predictions and creates results.md with images.
"""
import os
import random
from pathlib import Path

import hydra
import torch
import rootutils
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from loguru import logger

# Setup root directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.vit_model import ViTClassifier


def get_random_test_images(data_dir: str, num_images: int = 10):
    """Get random images from the dataset for inference."""
    data_path = Path(data_dir)
    all_images = []
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for img_path in class_dir.glob("*.jpg"):
                all_images.append((img_path, class_name))
    
    # Randomly sample images
    if len(all_images) < num_images:
        num_images = len(all_images)
    
    return random.sample(all_images, num_images)


def save_prediction_image(
    image_path: Path,
    prediction: str,
    target: str,
    confidence: float,
    save_path: Path
):
    """Save image with prediction annotation."""
    img = Image.open(image_path)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis('off')
    
    # Color based on correct/incorrect
    color = 'green' if prediction == target else 'red'
    title = f"Pred: {prediction} ({confidence:.2%})\nTarget: {target}"
    ax.set_title(title, fontsize=14, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def generate_results_md(predictions: list, output_dir: str):
    """Generate results.md with prediction images."""
    md_content = """# 🐱🐕 CatDog ViT Classifier Results

## Model Information
- **Model**: Vision Transformer (ViT)
- **Task**: Binary Classification (Cat vs Dog)
- **Framework**: PyTorch Lightning + timm

---

## Inference Results on Test Images

"""
    
    # Add each prediction
    for i, pred in enumerate(predictions):
        status = "✅" if pred["correct"] else "❌"
        md_content += f"""### Image {i+1} {status}

| Attribute | Value |
|-----------|-------|
| Prediction | **{pred['prediction']}** |
| Target | {pred['target']} |
| Confidence | {pred['confidence']:.2%} |

![Prediction {i+1}](predictions/pred_{i+1}.png)

---

"""
    
    # Add summary
    correct = sum(1 for p in predictions if p["correct"])
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0
    
    md_content += f"""## Summary

| Metric | Value |
|--------|-------|
| Total Images | {total} |
| Correct Predictions | {correct} |
| Accuracy | **{accuracy:.2%}** |

"""
    
    # Save results.md
    results_path = Path(output_dir) / "results.md"
    with open(results_path, 'w') as f:
        f.write(md_content)
    
    logger.info(f"Generated results.md at {results_path}")
    return results_path


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer_vit")
def main(cfg: DictConfig):
    """Main inference function."""
    logger.info("="*50)
    logger.info("Starting CatDog ViT Inference")
    logger.info("="*50)
    
    # Create output directories
    output_dir = cfg.output_dir
    predictions_dir = Path(output_dir) / "predictions"
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = cfg.checkpoint_path
    if not os.path.exists(checkpoint_path):
        # Try to find from best_model_path.txt
        best_path_file = Path(cfg.get("model_dir", "outputs")) / "best_model_path.txt"
        if best_path_file.exists():
            with open(best_path_file, 'r') as f:
                checkpoint_path = f.read().strip()
    
    logger.info(f"Loading model from: {checkpoint_path}")
    model = ViTClassifier.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    
    # Get class names
    class_names = model.hparams.class_names
    if not class_names:
        class_names = ["Cat", "Dog"]
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get random test images
    random.seed(cfg.seed)
    test_images = get_random_test_images(cfg.data_dir, cfg.num_images)
    
    predictions = []
    
    for i, (img_path, target) in enumerate(test_images):
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
        
        prediction = class_names[pred_idx]
        correct = prediction == target
        
        # Save annotated image
        save_path = predictions_dir / f"pred_{i+1}.png"
        save_prediction_image(img_path, prediction, target, confidence, save_path)
        
        predictions.append({
            "image": str(img_path.name),
            "prediction": prediction,
            "target": target,
            "confidence": confidence,
            "correct": correct,
        })
        
        logger.info(f"Image {i+1}: {prediction} ({confidence:.2%}) - Target: {target} - {'✓' if correct else '✗'}")
    
    # Generate results.md
    generate_results_md(predictions, output_dir)
    
    # Print summary
    correct = sum(1 for p in predictions if p["correct"])
    total = len(predictions)
    logger.info("="*50)
    logger.info(f"Inference complete: {correct}/{total} correct ({correct/total:.2%})")
    logger.info("="*50)
    
    return predictions


if __name__ == "__main__":
    main()
