"""
Inference script with Hydra configuration.
Runs predictions on images in a folder and saves results.
"""
import json
import os
from pathlib import Path
from typing import List

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from loguru import logger
from PIL import Image
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Setup root directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.train import DogBreedClassifier


def setup_logging(log_file: str = None):
    """Configure loguru logging."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(log_file, rotation="10 MB")


def process_image(image_path: Path, transform) -> tuple:
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0), image


def get_image_files(input_folder: str, extensions: List[str], max_images: int) -> List[Path]:
    """Get list of image files from input folder."""
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_folder).glob(f"*{ext}"))
        image_files.extend(Path(input_folder).glob(f"*{ext.upper()}"))
    return sorted(image_files)[:max_images]


def display_predictions(results: List[dict], original_images: List, output_folder: str, cols: int = 5):
    """Create and save visualization of predictions."""
    num_images = len(results)
    if num_images == 0:
        return
    
    rows = (num_images + cols - 1) // cols
    
    plt.figure(figsize=(20, 4 * rows))
    for idx, (result, img) in enumerate(zip(results, original_images)):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img)
        plt.axis('off')
        title = f"{result['predicted_class']}\n{result['confidence']}"
        plt.title(title, pad=10)
    
    plt.tight_layout()
    viz_path = os.path.join(output_folder, 'predictions_visualization.png')
    plt.savefig(viz_path)
    plt.close()
    logger.info(f"Saved predictions visualization to {viz_path}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer")
def main(cfg: DictConfig) -> List[dict]:
    """
    Main inference function.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        List of prediction results
    """
    setup_logging(log_file=f"{cfg.output_folder}/inference.log")
    
    logger.info("Starting inference process")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Input folder: {cfg.input_folder}")
    logger.info(f"Output folder: {cfg.output_folder}")
    logger.info(f"Checkpoint: {cfg.checkpoint_path}")

    # Create output directory
    os.makedirs(cfg.output_folder, exist_ok=True)

    # Resolve checkpoint path
    checkpoint_path = cfg.checkpoint_path
    if not os.path.exists(checkpoint_path):
        # Try to find best_model_path.txt
        best_path_file = Path(cfg.paths.root_dir) / "best_model_path.txt"
        if best_path_file.exists():
            with open(best_path_file, 'r') as f:
                checkpoint_path = f.read().strip()
            logger.info(f"Using checkpoint from best_model_path.txt: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DogBreedClassifier.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully (using {device})")
    
    # Get class mapping from model hyperparameters
    class_mapping = model.hparams.class_mapping
    if not class_mapping:
        logger.warning("No class mapping found in model hyperparameters, using indices")
        class_mapping = {i: str(i) for i in range(model.hparams.num_classes)}
    logger.info(f"Loaded class mapping with {len(class_mapping)} classes")

    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get image files
    image_files = get_image_files(
        cfg.input_folder, 
        list(cfg.image_extensions), 
        cfg.max_images
    )
    
    if not image_files:
        logger.error("No valid image files found in input folder")
        return []

    logger.info(f"Processing {len(image_files)} images")

    # Process images
    results = []
    original_images = []
    console = Console()
    
    for img_path in track(image_files, description="Processing images"):
        try:
            # Prepare image
            img_tensor, original_img = process_image(img_path, transform)
            img_tensor = img_tensor.to(device)
            original_images.append(original_img)

            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                pred_idx = probabilities.argmax().item()
                confidence = probabilities[pred_idx].item()

            # Get class name from the mapping
            pred_class = class_mapping.get(pred_idx, str(pred_idx))

            # Save result
            result = {
                'image': img_path.name,
                'predicted_class': pred_class,
                'confidence': f"{confidence:.2%}",
                'confidence_score': confidence
            }
            results.append(result)
            
            logger.debug(f"Processed {img_path.name}: {pred_class} ({confidence:.2%})")

        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
            continue

    # Save results to JSON
    output_file = Path(cfg.output_folder) / 'predictions.json'
    with open(output_file, 'w') as f:
        # Remove confidence_score for JSON output (keep only formatted string)
        json_results = [{k: v for k, v in r.items() if k != 'confidence_score'} for r in results]
        json.dump(json_results, f, indent=2)
    
    # Display results in a nice table
    table = Table(title="🐕 Inference Results")
    table.add_column("Image", style="cyan")
    table.add_column("Predicted Breed", style="green")
    table.add_column("Confidence", justify="right", style="yellow")

    for result in results:
        table.add_row(
            result['image'],
            result['predicted_class'],
            result['confidence']
        )

    console.print(table)
    logger.info(f"Results saved to {output_file}")

    # Create and save the visualization
    if cfg.save_visualization and results:
        display_predictions(results, original_images, cfg.output_folder, cfg.visualization_cols)

    logger.info("Inference complete!")
    return results


if __name__ == "__main__":
    main()
