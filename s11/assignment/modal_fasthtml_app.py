import modal
from typing import Annotated
import sys
import io
import os
import base64

# Define the Modal App
app = modal.App("cat-dog-classifier-fasthtml")

# Define the image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi==0.115.5",
        "onnxruntime==1.20.1",
        "numpy==1.26.4", # Using 1.x to avoid conflicts just in case
        "pillow==10.4.0",
        "pydantic==2.9.2",
        "python-fasthtml",
        "shad4fast",
        "sqlite-minutils",
        "uvicorn"
    )
    .add_local_file("model.onnx", "/app/model.onnx")
)

@app.function(
    image=image,
    # OnnxRuntime can use CPU quite efficiently for this small model, 
    # but we can add GPU if needed. T4 is fine.
    # gpu="T4", 
    scaledown_window=300,
)
@modal.asgi_app()
def web():
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from fastapi import FastAPI, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from fastapi.responses import JSONResponse, HTMLResponse
    
    # FastHTML imports
    from fasthtml.common import (
        Html, Script, Head, Title, Body, Div, Form, Input, Img, P, to_xml
    )
    from shad4fast import (
        ShadHead, Card, CardHeader, CardTitle, CardDescription, CardContent, 
        Alert, AlertTitle, AlertDescription, Button, Badge, Separator, Lucide, Progress
    )

    # Create main FastAPI app
    web_app = FastAPI(
        title="Image Classification API",
        description="FastAPI application serving an ONNX model for image classification",
        version="1.0.0",
    )

    # Add CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Model configuration
    INPUT_SIZE = (160, 160)
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    LABELS = ["Dog", "Cat"]
    MODEL_PATH = "/app/model.onnx"

    # Load the ONNX model
    try:
        print("Loading ONNX model...")
        ort_session = ort.InferenceSession(MODEL_PATH)
        # Warmup
        ort_session.run(
            ["output"], {"input": np.random.randn(1, 3, *INPUT_SIZE).astype(np.float32)}
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # We don't raise here to allow the app to start, but requests will fail
        ort_session = None

    class PredictionResponse(BaseModel):
        predictions: dict
        success: bool
        message: str

    def preprocess_image(image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        image = image.resize(INPUT_SIZE)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = (img_array - MEAN) / STD
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, 0)
        return img_array

    @web_app.get("/", response_class=HTMLResponse)
    async def ui_home():
        content = Html(
            Head(
                Title("Cat vs Dog Classifier"),
                ShadHead(tw_cdn=True, theme_handle=True),
                Script(
                    src="https://unpkg.com/htmx.org@2.0.3",
                    integrity="sha384-0895/pl2MU10Hqc6jd4RvrthNlDiE9U1tWmX7WRESftEDRosgxNsQG/Ze9YMRzHq",
                    crossorigin="anonymous",
                ),
            ),
            Body(
                Div(
                    Card(
                        CardHeader(
                            Div(
                                CardTitle("Cat vs Dog Classifier 🐱 🐶"),
                                Badge("AI Powered", variant="secondary", cls="w-fit"),
                                cls="flex items-center justify-between",
                            ),
                            CardDescription(
                                "Upload an image to classify whether it's a cat or a dog. Our AI model will analyze it instantly!"
                            ),
                        ),
                        CardContent(
                            Form(
                                Div(
                                    Div(
                                        Input(
                                            type="file",
                                            name="file",
                                            accept="image/*",
                                            required=True,
                                            cls="mb-4 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90 file:cursor-pointer",
                                        ),
                                        P(
                                            "Drag and drop an image or click to browse",
                                            cls="text-sm text-muted-foreground text-center mt-2",
                                        ),
                                        cls="border-2 border-dashed rounded-lg p-4 hover:border-primary/50 transition-colors",
                                    ),
                                    Button(
                                        Lucide("sparkles", cls="mr-2 h-4 w-4"),
                                        "Classify Image",
                                        type="submit",
                                        cls="w-full",
                                    ),
                                    cls="space-y-4",
                                ),
                                enctype="multipart/form-data",
                                hx_post="/classify",
                                hx_target="#result",
                            ),
                            Div(id="result", cls="mt-6"),
                        ),
                        cls="w-full max-w-3xl shadow-lg",
                        standard=True,
                    ),
                    cls="container flex items-center justify-center min-h-screen p-4",
                ),
                cls="bg-background text-foreground",
            ),
        )
        return to_xml(content)

    @web_app.post("/classify", response_class=HTMLResponse)
    async def ui_handle_classify(file: Annotated[bytes, File()]):
        if ort_session is None:
             return to_xml(Alert(AlertTitle("Error"), AlertDescription("Model not loaded"), variant="destructive"))

        try:
            # Predict Logic inline
            image = Image.open(io.BytesIO(file))
            processed_image = preprocess_image(image)
            outputs = ort_session.run(["output"], {"input": processed_image.astype(np.float32)})
            logits = outputs[0][0]
            probs_np = np.exp(logits) / np.sum(np.exp(logits))
            predictions = {LABELS[i]: float(prob) for i, prob in enumerate(probs_np)}
            
            # Formatting Response
            image_b64 = base64.b64encode(file).decode("utf-8")
            predicted_class = max(predictions.items(), key=lambda x: x[1])[0]
            confidence = max(predictions.values())
            emoji_map = {"Cat": "🐱", "Dog": "🐶"}

            results = Div(
                Div(
                    Div(
                        Img(
                            src=f"data:image/jpeg;base64,{image_b64}",
                            alt="Uploaded image",
                            cls="w-full rounded-lg shadow-lg aspect-square object-cover",
                        ),
                        cls="relative group",
                    ),
                    Div(
                        Badge(
                            f"It's a {predicted_class.lower()}! {emoji_map[predicted_class]}",
                            variant="outline",
                            cls=f"{'bg-green-500/20 hover:bg-green-500/20 border-green-500/50' if confidence > 0.8 else 'bg-yellow-500/20 hover:bg-yellow-500/20 border-yellow-500/50'} text-lg",
                        ),
                        Div(
                            Div(
                                P("Confidence Score", cls="font-medium"),
                                P(f"{confidence:.1%}", cls=f"text-xl font-bold"),
                                cls="flex justify-between items-baseline",
                            ),
                            Progress(value=int(confidence * 100), cls="h-2"),
                            cls="mt-4 space-y-2",
                        ),
                        Separator(cls="my-4"),
                        P("Detailed Analysis", cls="font-medium mb-2"),
                        Div(
                            *[
                                Div(
                                    Div(
                                        P(f"{label} {emoji_map[label]}", cls="font-medium"),
                                        P(f"{prob:.1%}", cls=f"font-medium { '' if label == predicted_class else 'text-muted-foreground'}"),
                                        cls="flex justify-between items-center",
                                    ),
                                    Progress(value=int(prob * 100), cls="h-2"),
                                    cls="space-y-2",
                                )
                                for label, prob in predictions.items()
                            ],
                            cls="space-y-4",
                        ),
                    ),
                    cls="grid grid-cols-1 md:grid-cols-2 gap-6",
                ),
                cls="animate-in fade-in-50 duration-500",
            )
            return to_xml(results)

        except Exception as e:
            error_alert = Alert(
                AlertTitle("Error ❌"),
                AlertDescription(str(e)),
                variant="destructive",
                cls="mt-4",
            )
            return to_xml(error_alert)

    # API Endpoint
    @web_app.post("/predict", response_model=PredictionResponse)
    async def predict_api(file: Annotated[bytes, File(description="Image file to classify")]):
        if ort_session is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        try:
            image = Image.open(io.BytesIO(file))
            processed_image = preprocess_image(image)
            outputs = ort_session.run(["output"], {"input": processed_image.astype(np.float32)})
            logits = outputs[0][0]
            probs_np = np.exp(logits) / np.sum(np.exp(logits))
            predictions = {LABELS[i]: float(prob) for i, prob in enumerate(probs_np)}
            return PredictionResponse(predictions=predictions, success=True, message="Classification successful")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    return web_app
