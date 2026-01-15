"""
TorchServe Handler for Z-Image-Turbo
Downloads model from HuggingFace at startup (instead of packaging in .mar)
"""

import logging
import os
import time
import numpy as np
import torch
from abc import ABC

logger = logging.getLogger(__name__)

class ZImageHandler(ABC):
    def __init__(self):
        self.initialized = False
        self.pipe = None
        self.device = None

    def initialize(self, ctx):
        """Load the model from HuggingFace Hub"""
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        
        # GPU Setup
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")
        
        # Import diffusers here (after TorchServe installs requirements)
        from diffusers import DiffusionPipeline
        
        # Model source: HuggingFace Hub (downloads at runtime)
        MODEL_ID = os.getenv("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
        
        # Check for local cache first
        local_cache = os.getenv("HF_HOME", "/home/model-server/hf_cache")
        
        logger.info(f"🚀 Loading model from HuggingFace: {MODEL_ID}")
        logger.info(f"📁 Cache directory: {local_cache}")
        
        start_time = time.time()
        
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir=local_cache
            )
            self.pipe.to(self.device)
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise e

    def preprocess(self, requests):
        """Extract prompts from requests"""
        inputs = []
        for idx, data in enumerate(requests):
            input_text = data.get("data") or data.get("body")
            
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            
            logger.info(f"📝 Prompt {idx}: {input_text[:100]}...")
            inputs.append(input_text)
        return inputs

    def inference(self, inputs):
        """Generate images"""
        logger.info(f"🎨 Generating {len(inputs)} image(s)...")
        
        try:
            images = self.pipe(
                inputs,
                num_inference_steps=50,
                guidance_scale=7.0
            ).images
            
            logger.info(f"✅ Generated {len(images)} image(s)")
            return images
            
        except Exception as e:
            logger.error(f"❌ Inference error: {e}")
            raise e

    def postprocess(self, inference_output):
        """Convert images to list format"""
        results = []
        for image in inference_output:
            results.append(np.array(image).tolist())
        return results

    def handle(self, data, context):
        """Main handler function"""
        if not self.initialized:
            self.initialize(context)
        
        inputs = self.preprocess(data)
        outputs = self.inference(inputs)
        return self.postprocess(outputs)
