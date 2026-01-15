import io
import socket
from contextlib import asynccontextmanager

import httpx
import timm
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

# Global model and labels
model = None
transform = None
labels = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, transform, labels, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model - using a pretrained ImageNet model
    model = timm.create_model("resnet50.a1_in1k", pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Get model specific transforms
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    # Load ImageNet labels
    url = "https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        labels = response.text.strip().split("\n")
    
    print("Model loaded successfully!")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Dog Breed Classifier API",
    description="A FastAPI service for classifying dog breeds using a pretrained model",
    version="1.0.0",
    lifespan=lifespan,
)

hostname = socket.gethostname()


def get_html_page():
    """Generate the FastHTML frontend page."""
    return f'''<!doctype html>
<html lang="en" style="margin: 0; padding: 0; min-height: 100%; display: flex; flex-direction: column;">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <title>Dog Breed Classifier</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        .container {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            max-width: 600px;
            width: 100%;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        h1 {{
            color: #e94560;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5rem;
            font-weight: 700;
        }}
        .pod-info {{
            color: #888;
            text-align: center;
            margin-bottom: 30px;
            font-size: 0.9rem;
        }}
        .upload-area {{
            border: 2px dashed #e94560;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        .upload-area:hover {{
            border-color: #ff6b6b;
            background: rgba(233, 69, 96, 0.1);
        }}
        .upload-area input[type="file"] {{
            display: none;
        }}
        .upload-area label {{
            color: #fff;
            cursor: pointer;
            font-size: 1.1rem;
        }}
        .upload-icon {{
            font-size: 3rem;
            margin-bottom: 15px;
            display: block;
        }}
        #preview {{
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            margin: 20px auto;
            display: none;
        }}
        button {{
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #e94560, #ff6b6b);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 1.1rem;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(233, 69, 96, 0.4);
        }}
        button:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }}
        #results {{
            margin-top: 30px;
            display: none;
        }}
        .result-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 15px;
            margin: 8px 0;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            color: #fff;
        }}
        .result-item:first-child {{
            background: rgba(233, 69, 96, 0.3);
            border: 1px solid #e94560;
        }}
        .label-name {{
            font-weight: 500;
        }}
        .confidence {{
            color: #e94560;
            font-weight: bold;
        }}
        .loading {{
            text-align: center;
            color: #fff;
            display: none;
        }}
        .spinner {{
            border: 3px solid rgba(255,255,255,0.1);
            border-top: 3px solid #e94560;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .api-link {{
            text-align: center;
            margin-top: 20px;
        }}
        .api-link a {{
            color: #e94560;
            text-decoration: none;
        }}
        .api-link a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🐕 Dog Breed Classifier</h1>
        <p class="pod-info">Pod: {hostname}</p>
        <div class="upload-area">
            <label for="file-input">
                <span class="upload-icon">📁</span>
                Click to upload an image or drag and drop
            </label>
            <input type="file" id="file-input" name="file" accept="image/*" onchange="previewImage(event)">
        </div>
        <img id="preview" alt="Preview">
        <button id="classify-btn" onclick="classifyImage()" disabled>Classify Image</button>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing image...</p>
        </div>
        <div id="results"></div>
        <div class="api-link">
            <a href="/docs" target="_blank">📚 API Documentation</a>
        </div>
    </div>
    <script>
        let selectedFile = null;
        
        function previewImage(event) {{
            const file = event.target.files[0];
            if (file) {{
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = function(e) {{
                    const preview = document.getElementById('preview');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    document.getElementById('classify-btn').disabled = false;
                }};
                reader.readAsDataURL(file);
            }}
        }}
        
        async function classifyImage() {{
            if (!selectedFile) return;
            
            const btn = document.getElementById('classify-btn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            
            btn.disabled = true;
            loading.style.display = 'block';
            results.style.display = 'none';
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            try {{
                const response = await fetch('/predict', {{
                    method: 'POST',
                    body: formData
                }});
                const data = await response.json();
                
                results.innerHTML = '<h3 style="color: #fff; margin-bottom: 15px;">Top Predictions:</h3>';
                data.predictions.forEach((pred, index) => {{
                    results.innerHTML += `
                        <div class="result-item">
                            <span class="label-name">${{index + 1}}. ${{pred.label}}</span>
                            <span class="confidence">${{(pred.confidence * 100).toFixed(2)}}%</span>
                        </div>
                    `;
                }});
                results.style.display = 'block';
            }} catch (error) {{
                results.innerHTML = '<p style="color: #ff6b6b;">Error classifying image. Please try again.</p>';
                results.style.display = 'block';
            }} finally {{
                loading.style.display = 'none';
                btn.disabled = false;
            }}
        }}
    </script>
</body>
</html>'''


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the FastHTML frontend."""
    return get_html_page()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded image.
    
    Returns top 5 predictions with confidence scores.
    """
    # Read and preprocess image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Transform and predict
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get top 5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    predictions = [
        {
            "label": labels[idx.item()].replace("_", " ").title(),
            "confidence": float(prob),
        }
        for prob, idx in zip(top5_prob, top5_idx)
    ]
    
    return {
        "predictions": predictions,
        "pod": hostname,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "pod": hostname}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
