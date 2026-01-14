import modal

# Create the Modal app
app = modal.App("cat-dog-classifier")

# Define the image with all dependencies and include the model
# Using Gradio 4.36.1 which is stable on Modal (4.0.0 is unstable)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision", 
        "gradio==4.36.1",
        "pillow",
        "huggingface_hub<0.25",
        "numpy",
    )
    .add_local_file("model.pt", "/app/model.pt")
)


@app.function(
    image=image,
    gpu="T4",
    scaledown_window=300,
)
@modal.asgi_app()
def web():
    """
    Cat-Dog Classifier Gradio App
    Using User's Logic with Stable Gradio Version
    """
    import gradio as gr
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np

    # Model configuration
    MODEL_PATH = "/app/model.pt"
    INPUT_SIZE = 224

    # Load model globally
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = torch.jit.load(MODEL_PATH, map_location=device)
    model = model.to(device)
    model.eval()
    print("Model loaded!")

    # Define transforms (EXACTLY AS PROVIDED)
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

        # Return exactly as provided in snippet
        return {"Cat": float(probs[0]), "Dog": float(probs[1])}

    # Create Gradio Blocks interface (EXACTLY AS PROVIDED)
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

    # Stability fixes for Modal
    if not hasattr(demo, "max_file_size"):
        demo.max_file_size = None

    demo.queue()
    return demo.app


@app.local_entrypoint()
def main():
    print("Cat-Dog Classifier ready!")
    print("Run 'modal serve modal_app.py' for development")
    print("Run 'modal deploy modal_app.py' for production")
