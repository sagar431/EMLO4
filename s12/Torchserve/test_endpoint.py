#!/usr/bin/env python3
"""
Test script for Z-Image-Turbo TorchServe endpoint
"""

import requests
import json
import numpy as np
from PIL import Image
import argparse
from datetime import datetime


def test_inference(prompt: str, output_path: str = None):
    """
    Send a prompt to TorchServe and save the generated image.
    
    Args:
        prompt: Text prompt for image generation
        output_path: Path to save the output image
    """
    url = "http://localhost:8080/predictions/z-image"
    
    print(f"📤 Sending request to TorchServe...")
    print(f"   Prompt: '{prompt}'")
    print(f"   URL: {url}")
    
    try:
        response = requests.post(url, data=prompt, timeout=600)
        
        if response.status_code == 200:
            print(f"✅ Response received! Status: {response.status_code}")
            
            # Parse response and construct image
            image_data = json.loads(response.text)
            image = Image.fromarray(np.array(image_data, dtype="uint8"))
            
            # Save image
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"torchserve_output_{timestamp}.png"
            
            image.save(output_path)
            print(f"🖼️  Image saved to: {output_path}")
            
        else:
            print(f"❌ Error! Status: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed! Is TorchServe running?")
        print("   Check with: curl http://localhost:8081/models")
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>10 minutes)")
    except Exception as e:
        print(f"❌ Error: {e}")


def check_model_status():
    """Check if the model is registered and its status."""
    url = "http://localhost:8081/models"
    
    print("🔍 Checking model status...")
    try:
        response = requests.get(url, timeout=10)
        print(f"   Models registered: {response.json()}")
    except Exception as e:
        print(f"❌ Could not connect to management API: {e}")


def check_health():
    """Check TorchServe health."""
    url = "http://localhost:8080/ping"
    
    print("🏥 Checking TorchServe health...")
    try:
        response = requests.get(url, timeout=10)
        print(f"   Status: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Z-Image TorchServe endpoint")
    parser.add_argument("--prompt", "-p", type=str, 
                        default="Astronaut riding a horse on Mars, detailed, 8k",
                        help="Text prompt for image generation")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for the image")
    parser.add_argument("--check", "-c", action="store_true",
                        help="Only check model status, don't generate")
    
    args = parser.parse_args()
    
    if args.check:
        check_health()
        check_model_status()
    else:
        check_health()
        check_model_status()
        print()
        test_inference(args.prompt, args.output)
