"""
FastAPI server for Z-Image-Turbo image generation
Connects to TorchServe backend and serves a web frontend
"""

import os
import io
import json
import base64
import requests
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration
TORCHSERVE_URL = os.getenv("TORCHSERVE_URL", "http://localhost:8080")
MODEL_NAME = os.getenv("MODEL_NAME", "z-image")

app = FastAPI(
    title="Z-Image-Turbo Generator",
    description="Generate stunning images using Z-Image-Turbo model via TorchServe",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
STATIC_DIR = Path("static")
GENERATED_DIR = Path("generated_images")
STATIC_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    
class GenerateResponse(BaseModel):
    success: bool
    image_base64: str = None
    image_path: str = None
    prompt: str = None
    error: str = None
    generation_time: float = None

class HealthResponse(BaseModel):
    status: str
    torchserve_status: str
    model_status: str

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page"""
    return get_frontend_html()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of FastAPI and TorchServe"""
    try:
        ts_response = requests.get(f"{TORCHSERVE_URL}/ping", timeout=5)
        ts_status = "healthy" if ts_response.status_code == 200 else "unhealthy"
        
        model_response = requests.get(f"http://localhost:8081/models/{MODEL_NAME}", timeout=5)
        if model_response.status_code == 200:
            model_data = model_response.json()
            workers = model_data[0].get("workers", [])
            ready_workers = [w for w in workers if w.get("status") == "READY"]
            model_status = f"ready ({len(ready_workers)} workers)"
        else:
            model_status = "not loaded"
            
    except Exception as e:
        ts_status = f"error: {str(e)}"
        model_status = "unknown"
    
    return HealthResponse(
        status="running",
        torchserve_status=ts_status,
        model_status=model_status
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """Generate an image from a text prompt"""
    import time
    start_time = time.time()
    
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    prompt = request.prompt.strip()
    
    try:
        response = requests.post(
            f"{TORCHSERVE_URL}/predictions/{MODEL_NAME}",
            data=prompt,
            timeout=600
        )
        
        if response.status_code != 200:
            return GenerateResponse(
                success=False,
                error=f"TorchServe error: {response.status_code} - {response.text}"
            )
        
        image_data = json.loads(response.text)
        image_array = np.array(image_data, dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"
        filepath = GENERATED_DIR / filename
        image.save(filepath)
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        generation_time = time.time() - start_time
        
        return GenerateResponse(
            success=True,
            image_base64=img_base64,
            image_path=str(filepath),
            prompt=prompt,
            generation_time=round(generation_time, 2)
        )
        
    except requests.exceptions.Timeout:
        return GenerateResponse(
            success=False,
            error="Request timed out. Image generation is taking too long."
        )
    except Exception as e:
        return GenerateResponse(
            success=False,
            error=str(e)
        )

@app.get("/gallery")
async def get_gallery():
    """Get list of previously generated images"""
    images = []
    for img_path in sorted(GENERATED_DIR.glob("*.png"), reverse=True)[:20]:
        with open(img_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()
        images.append({
            "filename": img_path.name,
            "path": str(img_path),
            "base64": img_base64,
            "created": img_path.stat().st_mtime
        })
    return {"images": images}


def get_frontend_html():
    """Return the frontend HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚡ Z-Image-Turbo Generator</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #050508;
            --bg-secondary: #0d0d14;
            --bg-tertiary: #151520;
            --bg-card: rgba(20, 20, 35, 0.8);
            --accent-1: #6366f1;
            --accent-2: #8b5cf6;
            --accent-3: #a855f7;
            --accent-4: #ec4899;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 25%, #a855f7 50%, #ec4899 100%);
            --text-primary: #ffffff;
            --text-secondary: #a1a1aa;
            --text-muted: #52525b;
            --border-color: rgba(99, 102, 241, 0.2);
            --glow: 0 0 40px rgba(139, 92, 246, 0.3);
            --success: #10b981;
            --error: #ef4444;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Animated gradient background */
        .bg-gradient {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            z-index: -1;
            background: 
                radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(168, 85, 247, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(236, 72, 153, 0.08) 0%, transparent 50%);
            animation: bgMove 20s ease-in-out infinite;
        }
        
        @keyframes bgMove {
            0%, 100% { transform: scale(1) rotate(0deg); }
            50% { transform: scale(1.1) rotate(3deg); }
        }
        
        /* Floating particles */
        .particles {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            z-index: -1;
            overflow: hidden;
        }
        
        .particle {
            position: absolute;
            width: 4px; height: 4px;
            background: var(--accent-2);
            border-radius: 50%;
            opacity: 0.3;
            animation: float 15s infinite ease-in-out;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(100vh) scale(0); opacity: 0; }
            10% { opacity: 0.3; }
            90% { opacity: 0.3; }
            100% { transform: translateY(-100vh) scale(1); opacity: 0; }
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 1.5rem;
        }
        
        /* Header */
        header {
            text-align: center;
            padding: 2rem 0;
        }
        
        .logo-container {
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }
        
        .logo-icon {
            font-size: 3rem;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        .logo {
            font-size: 3rem;
            font-weight: 800;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.03em;
        }
        
        .tagline {
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 400;
        }
        
        /* Status bar */
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            margin: 1.5rem 0;
            flex-wrap: wrap;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 1.2rem;
            background: var(--bg-card);
            border-radius: 50px;
            border: 1px solid var(--border-color);
            backdrop-filter: blur(10px);
            font-size: 0.9rem;
        }
        
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            animation: blink 2s infinite;
        }
        
        .status-dot.healthy { background: var(--success); box-shadow: 0 0 10px var(--success); }
        .status-dot.unhealthy { background: var(--error); }
        .status-dot.loading { background: #fbbf24; }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        
        /* Main layout */
        .main-layout {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-top: 1rem;
        }
        
        @media (max-width: 1024px) {
            .main-layout { grid-template-columns: 1fr; }
        }
        
        /* Generator section */
        .generator-section {
            background: var(--bg-card);
            border-radius: 24px;
            padding: 2rem;
            border: 1px solid var(--border-color);
            backdrop-filter: blur(20px);
        }
        
        .section-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .prompt-container {
            position: relative;
        }
        
        .prompt-input {
            width: 100%;
            min-height: 120px;
            padding: 1.25rem;
            font-size: 1rem;
            background: var(--bg-tertiary);
            border: 2px solid var(--border-color);
            border-radius: 16px;
            color: var(--text-primary);
            font-family: inherit;
            resize: vertical;
            transition: all 0.3s ease;
            line-height: 1.6;
        }
        
        .prompt-input:focus {
            outline: none;
            border-color: var(--accent-1);
            box-shadow: var(--glow);
        }
        
        .prompt-input::placeholder { color: var(--text-muted); }
        
        .char-count {
            position: absolute;
            bottom: 12px;
            right: 12px;
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        
        .generate-btn {
            width: 100%;
            padding: 1.25rem;
            margin-top: 1rem;
            font-size: 1.1rem;
            font-weight: 600;
            background: var(--accent-gradient);
            border: none;
            border-radius: 16px;
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            position: relative;
            overflow: hidden;
        }
        
        .generate-btn::before {
            content: '';
            position: absolute;
            top: 0; left: -100%;
            width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        
        .generate-btn:hover:not(:disabled)::before { left: 100%; }
        
        .generate-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 15px 50px rgba(139, 92, 246, 0.4);
        }
        
        .generate-btn:disabled {
            opacity: 0.7;
            cursor: not-allowed;
        }
        
        .spinner {
            width: 22px; height: 22px;
            border: 3px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        /* Random prompt suggestions */
        .suggestions-container {
            margin-top: 1.5rem;
        }
        
        .suggestions-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
        }
        
        .suggestions-title {
            font-size: 0.9rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .refresh-btn {
            background: none;
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.2s;
        }
        
        .refresh-btn:hover {
            border-color: var(--accent-1);
            color: var(--accent-1);
        }
        
        .suggestion-cards {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        
        .suggestion-card {
            padding: 1rem 1.25rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9rem;
            line-height: 1.5;
            color: var(--text-secondary);
            position: relative;
            overflow: hidden;
        }
        
        .suggestion-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0;
            width: 3px; height: 100%;
            background: var(--accent-gradient);
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .suggestion-card:hover {
            border-color: var(--accent-1);
            color: var(--text-primary);
            transform: translateX(5px);
        }
        
        .suggestion-card:hover::before { opacity: 1; }
        
        .suggestion-card.animate-in {
            animation: slideIn 0.4s ease forwards;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        /* Result section */
        .result-section {
            background: var(--bg-card);
            border-radius: 24px;
            padding: 2rem;
            border: 1px solid var(--border-color);
            backdrop-filter: blur(20px);
            min-height: 500px;
            display: flex;
            flex-direction: column;
        }
        
        .result-area {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            position: relative;
        }
        
        .placeholder {
            text-align: center;
            color: var(--text-muted);
        }
        
        .placeholder-icon {
            font-size: 5rem;
            margin-bottom: 1rem;
            animation: float-icon 3s ease-in-out infinite;
        }
        
        @keyframes float-icon {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        .generated-image {
            max-width: 100%;
            max-height: 450px;
            border-radius: 16px;
            box-shadow: 0 25px 80px rgba(0,0,0,0.6);
            transition: all 0.5s ease;
        }
        
        .generated-image.fade-in {
            animation: imageFadeIn 0.8s ease forwards;
        }
        
        @keyframes imageFadeIn {
            from { opacity: 0; transform: scale(0.9) translateY(20px); }
            to { opacity: 1; transform: scale(1) translateY(0); }
        }
        
        .result-info {
            margin-top: 1.5rem;
            text-align: center;
            padding: 1rem;
            background: var(--bg-tertiary);
            border-radius: 12px;
            width: 100%;
        }
        
        .result-prompt {
            color: var(--text-secondary);
            font-size: 0.95rem;
            line-height: 1.5;
            margin-bottom: 0.5rem;
        }
        
        .result-time {
            color: var(--success);
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        /* Loading animation */
        .loading-container {
            text-align: center;
        }
        
        .loading-orb {
            width: 100px; height: 100px;
            margin: 0 auto 1.5rem;
            position: relative;
        }
        
        .loading-orb::before, .loading-orb::after {
            content: '';
            position: absolute;
            width: 100%; height: 100%;
            border-radius: 50%;
            border: 3px solid transparent;
        }
        
        .loading-orb::before {
            border-top-color: var(--accent-1);
            border-right-color: var(--accent-2);
            animation: spin 1.5s linear infinite;
        }
        
        .loading-orb::after {
            width: 70%; height: 70%;
            top: 15%; left: 15%;
            border-top-color: var(--accent-3);
            border-left-color: var(--accent-4);
            animation: spin 1s linear infinite reverse;
        }
        
        .loading-text {
            color: var(--text-secondary);
            font-size: 1rem;
        }
        
        .loading-steps {
            color: var(--accent-2);
            font-weight: 500;
            margin-top: 0.5rem;
        }
        
        /* Gallery section */
        .gallery-section {
            margin-top: 2rem;
            background: var(--bg-card);
            border-radius: 24px;
            padding: 2rem;
            border: 1px solid var(--border-color);
            backdrop-filter: blur(20px);
        }
        
        .gallery-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
        }
        
        .gallery-scroll {
            display: flex;
            gap: 1rem;
            overflow-x: auto;
            padding-bottom: 1rem;
            scroll-behavior: smooth;
        }
        
        .gallery-scroll::-webkit-scrollbar {
            height: 6px;
        }
        
        .gallery-scroll::-webkit-scrollbar-track {
            background: var(--bg-tertiary);
            border-radius: 3px;
        }
        
        .gallery-scroll::-webkit-scrollbar-thumb {
            background: var(--accent-1);
            border-radius: 3px;
        }
        
        .gallery-item {
            flex-shrink: 0;
            width: 200px;
            height: 200px;
            border-radius: 12px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }
        
        .gallery-item:hover {
            transform: scale(1.05);
            border-color: var(--accent-1);
            box-shadow: var(--glow);
        }
        
        .gallery-item img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .gallery-item.fade-out {
            animation: fadeOut 0.5s ease forwards;
        }
        
        @keyframes fadeOut {
            to { opacity: 0; transform: scale(0.8); }
        }
        
        /* Error message */
        .error-message {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--error);
            color: var(--error);
            padding: 1.25rem;
            border-radius: 12px;
            text-align: center;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .container { padding: 1rem; }
            .logo { font-size: 2rem; }
            .logo-icon { font-size: 2rem; }
            .generator-section, .result-section { padding: 1.5rem; }
        }
    </style>
</head>
<body>
    <div class="bg-gradient"></div>
    <div class="particles" id="particles"></div>
    
    <div class="container">
        <header>
            <div class="logo-container">
                <span class="logo-icon">⚡</span>
                <h1 class="logo">Z-Image-Turbo</h1>
            </div>
            <p class="tagline">Transform your imagination into stunning AI art</p>
        </header>
        
        <div class="status-bar">
            <div class="status-item">
                <span class="status-dot loading" id="apiStatus"></span>
                <span id="apiStatusText">Connecting...</span>
            </div>
            <div class="status-item">
                <span class="status-dot loading" id="modelStatus"></span>
                <span id="modelStatusText">Loading model...</span>
            </div>
        </div>
        
        <div class="main-layout">
            <section class="generator-section">
                <h2 class="section-title">✨ Create Your Vision</h2>
                
                <div class="prompt-container">
                    <textarea class="prompt-input" 
                              id="promptInput" 
                              placeholder="Describe your dream image in vivid detail..."
                              oninput="updateCharCount()"></textarea>
                    <span class="char-count" id="charCount">0 / 500</span>
                </div>
                
                <button class="generate-btn" id="generateBtn" onclick="generateImage()">
                    <span id="btnIcon">🎨</span>
                    <span id="btnText">Generate Masterpiece</span>
                </button>
                
                <div class="suggestions-container">
                    <div class="suggestions-header">
                        <span class="suggestions-title">💡 Try these prompts</span>
                        <button class="refresh-btn" onclick="refreshSuggestions()">🔄 Shuffle</button>
                    </div>
                    <div class="suggestion-cards" id="suggestionCards"></div>
                </div>
            </section>
            
            <section class="result-section">
                <h2 class="section-title">🖼️ Your Creation</h2>
                <div class="result-area" id="resultArea">
                    <div class="placeholder" id="placeholder">
                        <div class="placeholder-icon">🌌</div>
                        <p>Your AI masterpiece will appear here</p>
                        <p style="margin-top: 0.5rem; font-size: 0.9rem;">Click a suggestion or write your own prompt</p>
                    </div>
                </div>
            </section>
        </div>
        
        <section class="gallery-section" id="gallerySection" style="display: none;">
            <div class="gallery-header">
                <h2 class="section-title">🎭 Recent Creations</h2>
            </div>
            <div class="gallery-scroll" id="galleryScroll"></div>
        </section>
    </div>
    
    <script>
        // Amazing prompts collection
        const allPrompts = [
            "Cinematic close-up of red grapes on a marble table, slanted shadows, soft diffused lighting, water droplets",
            "Renaissance style painting of a majestic stag in a forest clearing, rich oil colors, dramatic chiaroscuro lighting, painted in the style of Landseer",
            "Expressionist painting of a bioluminescent forest, bold brushstrokes creating ethereal glowing trees, floating spores of light swirling like stars",
            "Dramatic macro shot of a large garden spider in its web, morning dew drops glistening on silk strands, golden sunlight filtering through creating ethereal bokeh",
            "Expressionist painting of monumental 'EMLO4' letters on Mars, bold impasto brushstrokes, lone astronaut dwarfed by metallic letters, dramatic swirling cosmos",
            "Minimalist line art of a blooming cherry tree branch, continuous single line drawing, delicate petals flowing into each other, white on black",
            "Portrait photography of a young woman in natural morning light, freckled face, candid expression, wearing oversized knit sweater, medium format film",
            "Portrait of an elderly man with weathered face, Rembrandt lighting, black and white",
            "Watercolor painting of floating islands in sunset sky, soft edges blending with clouds, waterfalls flowing upwards creating misty rainbows",
            "Renaissance style painting of a modern coffee shop interior, rays of sunlight through windows",
            "Massive ice sculpture of a dragon, light passing through creates rainbow prisms, ethereal mist surrounding the base",
            "Renaissance style painting of an underwater garden, classical marble ruins overtaken by coral, shafts of light piercing through water",
            "Retro poster design for 'The School of AI', neon lighting effects, cyberpunk style",
            "Renaissance style painting of a giant monarch butterfly in a classical garden, golden hour light, marble columns with ivy",
            "Ancient tree city with branches forming natural bridges, leaves made of stained glass, sunset light filtering through, Studio Ghibli style",
            "Minimalist line art of a floating observatory among stars, single continuous line drawing style, delicate geometric patterns, white on black",
            "A cyberpunk samurai standing in neon-lit Tokyo rain, reflections on wet pavement, cinematic composition",
            "Ethereal forest spirit emerging from morning mist, translucent form with glowing particles, mystical atmosphere",
            "Steampunk airship battle above Victorian London, dramatic clouds, brass and copper details gleaming",
            "Surreal desert with melting clocks and floating geometric shapes, Salvador Dali inspired, dreamlike quality"
        ];
        
        let currentSuggestions = [];
        let generatedImages = [];
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            createParticles();
            checkStatus();
            refreshSuggestions();
            loadGallery();
            setInterval(checkStatus, 30000);
            setInterval(autoRefreshSuggestion, 8000); // Auto refresh one suggestion every 8s
        });
        
        // Create floating particles
        function createParticles() {
            const container = document.getElementById('particles');
            for (let i = 0; i < 30; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 15 + 's';
                particle.style.animationDuration = (10 + Math.random() * 10) + 's';
                container.appendChild(particle);
            }
        }
        
        // Check status
        async function checkStatus() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                
                const apiDot = document.getElementById('apiStatus');
                const apiText = document.getElementById('apiStatusText');
                const modelDot = document.getElementById('modelStatus');
                const modelText = document.getElementById('modelStatusText');
                
                if (data.torchserve_status === 'healthy') {
                    apiDot.className = 'status-dot healthy';
                    apiText.textContent = 'TorchServe Online';
                } else {
                    apiDot.className = 'status-dot unhealthy';
                    apiText.textContent = 'TorchServe Offline';
                }
                
                if (data.model_status.includes('ready')) {
                    modelDot.className = 'status-dot healthy';
                    modelText.textContent = 'Model Ready';
                } else {
                    modelDot.className = 'status-dot unhealthy';
                    modelText.textContent = 'Model ' + data.model_status;
                }
            } catch (e) {
                console.error('Status check failed:', e);
            }
        }
        
        // Refresh all suggestions
        function refreshSuggestions() {
            const shuffled = [...allPrompts].sort(() => Math.random() - 0.5);
            currentSuggestions = shuffled.slice(0, 3);
            renderSuggestions();
        }
        
        // Auto refresh one suggestion with animation
        function autoRefreshSuggestion() {
            const index = Math.floor(Math.random() * currentSuggestions.length);
            const availablePrompts = allPrompts.filter(p => !currentSuggestions.includes(p));
            if (availablePrompts.length > 0) {
                const newPrompt = availablePrompts[Math.floor(Math.random() * availablePrompts.length)];
                currentSuggestions[index] = newPrompt;
                renderSuggestions(index);
            }
        }
        
        // Render suggestions
        function renderSuggestions(animateIndex = -1) {
            const container = document.getElementById('suggestionCards');
            container.innerHTML = currentSuggestions.map((prompt, i) => `
                <div class="suggestion-card ${i === animateIndex ? 'animate-in' : ''}" 
                     onclick="usePrompt('${prompt.replace(/'/g, "\\'")}')">
                    ${prompt}
                </div>
            `).join('');
        }
        
        // Use prompt suggestion
        function usePrompt(text) {
            document.getElementById('promptInput').value = text;
            updateCharCount();
        }
        
        // Update character count
        function updateCharCount() {
            const input = document.getElementById('promptInput');
            const count = document.getElementById('charCount');
            count.textContent = `${input.value.length} / 500`;
        }
        
        // Generate image
        async function generateImage() {
            const promptInput = document.getElementById('promptInput');
            const generateBtn = document.getElementById('generateBtn');
            const btnIcon = document.getElementById('btnIcon');
            const btnText = document.getElementById('btnText');
            const resultArea = document.getElementById('resultArea');
            
            const prompt = promptInput.value.trim();
            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }
            
            // Show loading
            generateBtn.disabled = true;
            btnIcon.innerHTML = '<div class="spinner"></div>';
            btnText.textContent = 'Creating magic...';
            
            resultArea.innerHTML = `
                <div class="loading-container">
                    <div class="loading-orb"></div>
                    <p class="loading-text">Crafting your masterpiece...</p>
                    <p class="loading-steps">This takes ~30-60 seconds</p>
                </div>
            `;
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    resultArea.innerHTML = `
                        <img class="generated-image fade-in" 
                             src="data:image/png;base64,${data.image_base64}" 
                             alt="${data.prompt}">
                        <div class="result-info">
                            <p class="result-prompt">"${data.prompt}"</p>
                            <p class="result-time">✨ Generated in ${data.generation_time}s</p>
                        </div>
                    `;
                    
                    // Add to gallery
                    addToGallery(data.image_base64, data.prompt);
                } else {
                    resultArea.innerHTML = `
                        <div class="error-message">
                            <strong>❌ Error:</strong> ${data.error}
                        </div>
                    `;
                }
            } catch (e) {
                resultArea.innerHTML = `
                    <div class="error-message">
                        <strong>❌ Error:</strong> ${e.message}
                    </div>
                `;
            } finally {
                generateBtn.disabled = false;
                btnIcon.textContent = '🎨';
                btnText.textContent = 'Generate Masterpiece';
            }
        }
        
        // Add to gallery
        function addToGallery(base64, prompt) {
            const gallerySection = document.getElementById('gallerySection');
            const galleryScroll = document.getElementById('galleryScroll');
            
            gallerySection.style.display = 'block';
            
            const item = document.createElement('div');
            item.className = 'gallery-item';
            item.innerHTML = `<img src="data:image/png;base64,${base64}" alt="${prompt}">`;
            item.onclick = () => showInResult(base64, prompt);
            
            galleryScroll.insertBefore(item, galleryScroll.firstChild);
            
            // Fade out old items if too many
            const items = galleryScroll.querySelectorAll('.gallery-item');
            if (items.length > 10) {
                const lastItem = items[items.length - 1];
                lastItem.classList.add('fade-out');
                setTimeout(() => lastItem.remove(), 500);
            }
        }
        
        // Show image in result area
        function showInResult(base64, prompt) {
            const resultArea = document.getElementById('resultArea');
            resultArea.innerHTML = `
                <img class="generated-image fade-in" 
                     src="data:image/png;base64,${base64}" 
                     alt="${prompt}">
                <div class="result-info">
                    <p class="result-prompt">"${prompt}"</p>
                </div>
            `;
        }
        
        // Load existing gallery
        async function loadGallery() {
            try {
                const response = await fetch('/gallery');
                const data = await response.json();
                
                if (data.images && data.images.length > 0) {
                    const gallerySection = document.getElementById('gallerySection');
                    const galleryScroll = document.getElementById('galleryScroll');
                    
                    gallerySection.style.display = 'block';
                    
                    data.images.slice(0, 10).forEach(img => {
                        const item = document.createElement('div');
                        item.className = 'gallery-item';
                        item.innerHTML = `<img src="data:image/png;base64,${img.base64}" alt="Generated image">`;
                        item.onclick = () => {
                            const resultArea = document.getElementById('resultArea');
                            resultArea.innerHTML = `
                                <img class="generated-image fade-in" 
                                     src="data:image/png;base64,${img.base64}">
                            `;
                        };
                        galleryScroll.appendChild(item);
                    });
                }
            } catch (e) {
                console.error('Failed to load gallery:', e);
            }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
