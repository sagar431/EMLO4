import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from torchvision import transforms
from src.train import DogBreedClassifier
from loguru import logger
from PIL import Image
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from src.utils.logging_config import setup_logging
from rich.progress import track

def process_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0), image

def display_predictions(results, original_images, output_folder):
    num_images = len(results)
    cols = 5
    rows = (num_images + cols - 1) // cols
    
    plt.figure(figsize=(20, 4 * rows))
    for idx, (result, img) in enumerate(zip(results, original_images)):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img)
        plt.axis('off')
        title = f"{result['predicted_class']}\n{result['confidence']}"
        plt.title(title, pad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'predictions_visualization.png'))
    plt.close()
    logger.info(f"Saved predictions visualization to {output_folder}/predictions_visualization.png")

def main():
    setup_logging(log_file="logs/inference.log")
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_folder', type=str, required=True)
    parser.add_argument('-output_folder', type=str, required=True)
    parser.add_argument('-ckpt', type=str, required=True)
    args = parser.parse_args()

    logger.info("Starting inference process")
    logger.info(f"Input folder: {args.input_folder}")
    logger.info(f"Output folder: {args.output_folder}")
    logger.info(f"Checkpoint: {args.ckpt}")

    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DogBreedClassifier.load_from_checkpoint(args.ckpt)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully (using {device})")
    
    # Get class mapping from model hyperparameters
    class_mapping = model.hparams.class_mapping
    if not class_mapping:
        logger.error("No class mapping found in model hyperparameters")
        return
    logger.info(f"Loaded class mapping: {class_mapping}")

    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get image files
    image_files = [f for f in Path(args.input_folder).glob('*') 
                   if f.suffix.lower() in ['.jpg', '.jpeg', '.png']][:10]
    
    if not image_files:
        logger.error("No valid image files found in input folder")
        return

    logger.info(f"Processing {len(image_files)} images")

    # Process images
    results = []
    original_images = []
    console = Console()
    
    with console.status("[bold green]Processing images...") as status:
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
                pred_class = class_mapping[pred_idx]

                # Save result
                result = {
                    'image': img_path.name,
                    'predicted_class': pred_class,
                    'confidence': f"{confidence:.2%}"
                }
                results.append(result)
                
                logger.info(f"Processed {img_path.name}: {pred_class} ({confidence:.2%})")

            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue

    # Save results
    output_file = Path(args.output_folder) / 'predictions.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display results in a nice table
    table = Table(title="Inference Results")
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
    display_predictions(results, original_images, args.output_folder)

if __name__ == "__main__":
    main()
