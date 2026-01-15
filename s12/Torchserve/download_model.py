#!/usr/bin/env python3
import torch
from diffusers import DiffusionPipeline
import os

print("⏳ Downloading Z-Image-Turbo model...")

# Load the model
pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", 
    torch_dtype=torch.bfloat16
)

# Save to local directory
output_dir = "./z-image-model"
os.makedirs(output_dir, exist_ok=True)
pipe.save_pretrained(output_dir)

print(f"✅ Model saved to {output_dir}")
