"""
Cat-Dog Classifier Gradio App for HuggingFace Spaces
"""

import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image

# Model configuration
MODEL_PATH = "model.pt"
INPUT_SIZE = 224

# Load model globally
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = torch.jit.load(MODEL_PATH, map_location=device)
model = model.to(device)
model.eval()
print("Model loaded!")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict(image):
    """Predict whether the image is a cat or dog."""
    if image is None:
        return {"Cat": 0.0, "Dog": 0.0}

    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert('RGB')
    else:
        image = image.convert('RGB')

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    return {"Cat": float(probs[0]), "Dog": float(probs[1])}


# Create Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown("# Cat vs Dog Classifier")
    gr.Markdown("Upload an image of a cat or dog and the AI will predict which one it is.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Classify")
        with gr.Column():
            output_label = gr.Label(label="Prediction")

    submit_btn.click(fn=predict, inputs=input_image, outputs=output_label)

if __name__ == "__main__":
    demo.launch()
