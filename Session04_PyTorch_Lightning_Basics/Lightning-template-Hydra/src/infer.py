import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models.catdog_classifier import CatDogClassifier
from utils.rich_utils import setup_logger, create_rich_progress_bar

def process_image(image_path, transform):
    """Process a single image for inference"""
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description='Inference for Cat/Dog Classification')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder containing images')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for predictions')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger()
    logger.info("Starting inference...")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {args.ckpt_path}")
    model = CatDogClassifier.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get list of images
    image_paths = list(Path(args.input_folder).glob('*.[jp][pn][g]'))
    
    # Setup progress bar
    progress = create_rich_progress_bar()
    
    # Process images
    results = []
    with progress:
        task = progress.add_task("[cyan]Processing images...", total=len(image_paths))
        
        for img_path in image_paths:
            try:
                # Process image
                img_tensor = process_image(img_path, transform)
                img_tensor = img_tensor.to(model.device)

                # Inference
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = F.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()

                # Save results
                class_labels = ['cat', 'dog']
                prediction = {
                    'image': str(img_path),
                    'predicted_class': class_labels[predicted_class],
                    'confidence': f"{confidence:.4f}"
                }
                results.append(prediction)
                
                # Save to output file
                output_path = Path(args.output_folder) / f"{img_path.stem}_prediction.txt"
                with open(output_path, 'w') as f:
                    f.write(f"Image: {img_path.name}\n")
                    f.write(f"Predicted Class: {class_labels[predicted_class]}\n")
                    f.write(f"Confidence: {confidence:.4f}\n")
                
                logger.info(f"Processed {img_path.name}: {class_labels[predicted_class]} ({confidence:.4f})")
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
            
            progress.update(task, advance=1)

    logger.info("Inference completed!")
    logger.info(f"Results saved in {args.output_folder}")

if __name__ == "__main__":
    main()
