import io
import json
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger
from PIL import Image
import lightning as L
from rich.console import Console
from rich.table import Table

from src.train import DogBreedClassifier

app = FastAPI(title="Dog Breed Classifier API")
console = Console()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
CHECKPOINT_PATH = Path("checkpoints/dog-breed-epoch=05-val_acc=1.00.ckpt")
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global model
    logger.info(f"Loading model from {CHECKPOINT_PATH}")
    model = DogBreedClassifier.load_from_checkpoint(CHECKPOINT_PATH)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

@app.on_event("startup")
async def startup_event():
    load_model()

def process_image(image: Image.Image) -> torch.Tensor:
    # Resize and convert to tensor
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = torch.tensor(list(image.getdata())).reshape(1, 3, 224, 224).float() / 255.0
    return image.to(device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and process the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    tensor = process_image(image)
    
    # Make prediction
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
    
    # Get the predicted breed
    breed = model.hparams.class_mapping[prediction.item()]
    confidence = confidence.item() * 100
    
    # Create response
    result = {
        "filename": file.filename,
        "predicted_breed": breed,
        "confidence": f"{confidence:.2f}%"
    }
    
    # Display result in console
    table = Table(title="Prediction Result")
    table.add_column("Image")
    table.add_column("Predicted Breed")
    table.add_column("Confidence")
    table.add_row(file.filename, breed, f"{confidence:.2f}%")
    console.print(table)
    
    return result

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dog Breed Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .upload-form {
                text-align: center;
                margin: 20px 0;
            }
            #result {
                margin-top: 20px;
                padding: 10px;
                border-radius: 4px;
            }
            .prediction {
                margin: 10px 0;
                padding: 15px;
                background-color: #e8f5e9;
                border-radius: 4px;
            }
            #imagePreview {
                max-width: 300px;
                margin: 10px auto;
                display: none;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐕 Dog Breed Classifier</h1>
            <div class="upload-form">
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
                <button onclick="document.getElementById('fileInput').click()">Choose Image</button>
                <button onclick="uploadImage()" id="uploadButton" disabled>Classify Dog Breed</button>
            </div>
            <img id="imagePreview">
            <div id="result"></div>
        </div>

        <script>
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('imagePreview').src = e.target.result;
                        document.getElementById('imagePreview').style.display = 'block';
                        document.getElementById('uploadButton').disabled = false;
                    }
                    reader.readAsDataURL(file);
                }
            });

            async function uploadImage() {
                const fileInput = document.getElementById('fileInput');
                const resultDiv = document.getElementById('result');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select an image first!');
                    return;
                }

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    
                    resultDiv.innerHTML = `
                        <div class="prediction">
                            <h3>Results:</h3>
                            <p><strong>Predicted Breed:</strong> ${result.predicted_breed}</p>
                            <p><strong>Confidence:</strong> ${result.confidence}</p>
                        </div>
                    `;
                } catch (error) {
                    resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """
