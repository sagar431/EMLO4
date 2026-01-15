# -*- coding: utf-8 -*-
"""Z-Image-Turbo.py

Local Inference on GPU for Z-Image-Turbo model
Model page: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
"""

import torch
from diffusers import DiffusionPipeline
import os
from datetime import datetime


def generate_image_local(prompt: str, output_path: str = None) -> str:
    """
    Generate an image using local GPU inference.
    
    Args:
        prompt: Text prompt for image generation
        output_path: Optional path to save the image. If None, auto-generates filename.
    
    Returns:
        Path to the saved image
    """
    # Load the pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", 
        torch_dtype=torch.bfloat16, 
        device_map="cuda"
    )
    
    # Generate image
    image = pipe(prompt).images[0]
    
    # Save the image
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output_{timestamp}.png"
    
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    return output_path


def generate_image_remote(prompt: str, hf_token: str = None, output_path: str = None) -> str:
    """
    Generate an image using HuggingFace Inference Providers (remote API).
    
    Args:
        prompt: Text prompt for image generation
        hf_token: HuggingFace API token. If None, uses HF_TOKEN env variable.
        output_path: Optional path to save the image. If None, auto-generates filename.
    
    Returns:
        Path to the saved image
    """
    from huggingface_hub import InferenceClient
    
    # Get token from argument or environment
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not provided. Set it as environment variable or pass as argument.")
    
    client = InferenceClient(
        provider="auto",
        api_key=token,
    )
    
    # Generate image via remote API
    image = client.text_to_image(
        prompt,
        model="Tongyi-MAI/Z-Image-Turbo",
    )
    
    # Save the image
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output_remote_{timestamp}.png"
    
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate images using Z-Image-Turbo model")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for the image")
    parser.add_argument("--mode", "-m", type=str, choices=["local", "remote"], default="local",
                        help="Inference mode: 'local' for GPU inference, 'remote' for HF API")
    parser.add_argument("--token", "-t", type=str, default=None, help="HuggingFace API token (for remote mode)")
    
    args = parser.parse_args()
    
    if args.mode == "local":
        generate_image_local(args.prompt, args.output)
    else:
        generate_image_remote(args.prompt, args.token, args.output)
