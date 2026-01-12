#!/usr/bin/env python3
"""
Script to trace/script the trained Cat-Dog classifier model and save it.
This creates a TorchScript model that can be deployed without the original Python code.

Usage:
    python src/trace_model.py --ckpt_path <path_to_checkpoint> --output_path <output_model.pt>

Example:
    python src/trace_model.py --ckpt_path logs/train/runs/2024-11-15/checkpoints/last.ckpt --output_path model.pt
"""

import os
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import timm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class CatDogClassifier(nn.Module):
    """
    Simple Cat-Dog classifier using timm backbone.
    This class structure should match your trained model.
    """
    def __init__(self, model_name: str = "resnet18", num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_model_from_checkpoint(ckpt_path: str, model_name: str = "resnet18", num_classes: int = 2) -> nn.Module:
    """
    Load model from Lightning checkpoint.
    
    Args:
        ckpt_path: Path to the checkpoint file
        model_name: timm model name
        num_classes: Number of output classes
    
    Returns:
        Loaded model in eval mode
    """
    log.info(f"Loading checkpoint from: {ckpt_path}")
    
    # Create model
    model = CatDogClassifier(model_name=model_name, num_classes=num_classes, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present (from Lightning)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # Remove 'model.' prefix
            elif k.startswith('net.'):
                new_state_dict[k[4:]] = v  # Remove 'net.' prefix
            else:
                new_state_dict[k] = v
        model.model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    log.info("Model loaded successfully!")
    return model


def trace_model(model: nn.Module, input_shape: tuple = (1, 3, 224, 224), method: str = "trace") -> torch.jit.ScriptModule:
    """
    Trace or script the model to create a TorchScript version.
    
    Args:
        model: PyTorch model to trace
        input_shape: Shape of input tensor (batch, channels, height, width)
        method: 'trace' or 'script'
    
    Returns:
        TorchScript model
    """
    model.eval()
    
    # Create example input
    example_input = torch.randn(*input_shape)
    
    if method == "trace":
        log.info(f"Tracing model with input shape: {input_shape}")
        # Trace the model - records operations during forward pass
        traced_model = torch.jit.trace(model, example_input)
    elif method == "script":
        log.info("Scripting model...")
        # Script the model - parses Python code
        traced_model = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")
    
    # Verify the traced model works
    log.info("Verifying traced model...")
    with torch.no_grad():
        original_output = model(example_input)
        traced_output = traced_model(example_input)
        
        # Check outputs match
        if torch.allclose(original_output, traced_output, rtol=1e-4, atol=1e-4):
            log.info("✅ Traced model outputs match original model!")
        else:
            log.warning("⚠️ Traced model outputs differ slightly from original")
    
    return traced_model


def save_traced_model(traced_model: torch.jit.ScriptModule, output_path: str):
    """Save the traced model to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.jit.save(traced_model, output_path)
    log.info(f"✅ Traced model saved to: {output_path}")
    
    # Print file size
    file_size = output_path.stat().st_size / (1024 * 1024)
    log.info(f"   Model size: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Trace/Script a trained model for deployment")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--output_path", type=str, default="model.pt", help="Output path for traced model")
    parser.add_argument("--model_name", type=str, default="resnet18", help="timm model name")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument("--method", type=str, default="trace", choices=["trace", "script"], help="Tracing method")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights (no checkpoint needed)")

    args = parser.parse_args()

    # Validate args
    if not args.pretrained and not args.ckpt_path:
        parser.error("Either --ckpt_path or --pretrained must be specified")

    log.info("=" * 60)
    log.info("🔧 Model Tracing Script")
    log.info("=" * 60)
    log.info(f"Checkpoint: {args.ckpt_path if args.ckpt_path else 'None (using pretrained)'}")
    log.info(f"Output: {args.output_path}")
    log.info(f"Model: {args.model_name}")
    log.info(f"Classes: {args.num_classes}")
    log.info(f"Input size: {args.input_size}x{args.input_size}")
    log.info(f"Method: {args.method}")
    log.info(f"Pretrained: {args.pretrained}")
    log.info("=" * 60)

    # Load model
    if args.pretrained:
        log.info("Creating model with ImageNet pretrained weights...")
        model = CatDogClassifier(model_name=args.model_name, num_classes=args.num_classes, pretrained=True)
        model.eval()
        log.info("Model created successfully!")
    else:
        model = load_model_from_checkpoint(
            ckpt_path=args.ckpt_path,
            model_name=args.model_name,
            num_classes=args.num_classes
        )
    
    # Trace model
    input_shape = (1, 3, args.input_size, args.input_size)
    traced_model = trace_model(model, input_shape=input_shape, method=args.method)
    
    # Save traced model
    save_traced_model(traced_model, args.output_path)
    
    log.info("=" * 60)
    log.info("✅ Model tracing complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
