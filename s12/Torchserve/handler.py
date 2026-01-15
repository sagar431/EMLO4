import logging
import zipfile
import time
import os
from abc import ABC
import numpy as np
import torch
from diffusers import DiffusionPipeline
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class ZImageHandler(BaseHandler, ABC):
    def __init__(self):
        self.initialized = False
        self.pipe = None

    def initialize(self, ctx):
        """Load the model and initialize the pipeline"""
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        
        # GPU Setup
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        # WEIGHTS LOADING STRATEGY
        # 1. Check if weights are mounted externally (Fastest, avoids unzip)
        external_weights = "/home/model-server/weights"
        
        if os.path.exists(external_weights) and os.path.exists(os.path.join(external_weights, "model_index.json")):
            logger.info(f"✨ Loading weights from external mount: {external_weights}")
            model_path = external_weights
            
        else:
            # 2. Fallback to extracting from zip (Standard TorchServe)
            logger.info("External weights not found, checking local archive...")
            zip_path = os.path.join(model_dir, "z-image-model.zip")
            extract_path = os.path.join(model_dir, "model")
            
            if os.path.exists(zip_path):
                logger.info(f"Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_path)
                model_path = extract_path
            else:
                model_path = os.path.join(model_dir, "z-image-model")

        logger.info(f"Loading pipeline from {model_path}")

        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16
            )
            self.pipe.to(self.device)
            self.initialized = True
            logger.info("✅ Z-Image-Turbo pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise e

    def preprocess(self, requests):
        """Clean and extract prompts from requests"""
        inputs = []
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            
            logger.info(f"Processing prompt {idx}: {input_text}")
            inputs.append(input_text)
        return inputs

    def inference(self, inputs):
        """Run generation"""
        logger.info(f"Generating images for {len(inputs)} prompts")
        try:
            # Z-Image-Turbo parameters
            inferences = self.pipe(
                inputs,
                num_inference_steps=50,  # Recommended 50 for this model
                guidance_scale=7.0
            ).images
            return inferences
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e

    def postprocess(self, inference_output):
        """Convert images to list for response"""
        images = []
        for image in inference_output:
            images.append(np.array(image).tolist())
        return images
