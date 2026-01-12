"""
Cat-Dog Classifier Gradio App for HuggingFace Spaces
"""

import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Class labels
LABELS = ['Cat', 'Dog']

# Model configuration
MODEL_PATH = "model.pt"
INPUT_SIZE = 224


class CatDogClassifier:
    """Cat-Dog Classifier using TorchScript traced model."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load the traced model
        print(f"Loading model from: {model_path}")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def predict(self, image):
        """Predict whether the image is a cat or dog."""
        if image is None:
            return "Please upload an image"

        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        else:
            image = image.convert('RGB')

        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get prediction
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get results
        cat_prob = float(probabilities[0]) * 100
        dog_prob = float(probabilities[1]) * 100

        prediction = "Cat" if cat_prob > dog_prob else "Dog"
        confidence = max(cat_prob, dog_prob)

        return f"{prediction} ({confidence:.1f}% confident)\n\nCat: {cat_prob:.1f}%\nDog: {dog_prob:.1f}%"


# Initialize classifier
print("Initializing Cat-Dog Classifier...")
classifier = CatDogClassifier()

# Create simple Gradio interface
demo = gr.Interface(
    fn=classifier.predict,
    inputs=gr.Image(label="Upload an image of a cat or dog"),
    outputs=gr.Textbox(label="Prediction"),
    title="Cat vs Dog Classifier",
    description="Upload an image of a cat or dog and the AI will predict which one it is.",
    examples=[
        ["examples/cat.jpg"],
        ["examples/dog.jpg"]
    ] if os.path.exists("examples") else None,
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()
