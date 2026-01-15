"""
TorchServe Handler for Z-Image-Turbo - OPTIMIZED VERSION
Optimizations applied:
1. Reduced inference steps (50 -> 25)
2. torch.compile() for PyTorch 2.0+ speedup
3. CUDA optimizations (cudnn benchmark, TF32)
4. Memory efficient attention (if available)
5. VAE slicing for memory optimization
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
        """Load and optimize the model"""
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        
        # GPU Setup
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        logger.info(f"🔧 Using device: {self.device}")
        
        # ============ OPTIMIZATION 1: CUDA Settings ============
        if torch.cuda.is_available():
            # Enable cuDNN benchmark for faster convolutions
            torch.backends.cudnn.benchmark = True
            # Enable TF32 for faster matrix multiplications (Ampere+ GPUs)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✅ CUDA optimizations enabled (cudnn.benchmark, TF32)")
        
        # Import diffusers
        from diffusers import DiffusionPipeline
        
        # Model source
        MODEL_ID = os.getenv("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
        local_cache = os.getenv("HF_HOME", "/home/model-server/hf_cache")
        
        # Check for local weights mount first
        local_weights = "/home/model-server/weights"
        if os.path.exists(local_weights) and os.path.exists(os.path.join(local_weights, "model_index.json")):
            model_path = local_weights
            logger.info(f"📁 Loading from local mount: {model_path}")
        else:
            model_path = MODEL_ID
            logger.info(f"🌐 Loading from HuggingFace: {model_path}")
        
        start_time = time.time()
        
        try:
            # ============ OPTIMIZATION 2: Use bfloat16 ============
            # bfloat16 has better numeric stability for this model
            self.pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,  # bfloat16 for stability
                cache_dir=local_cache if model_path == MODEL_ID else None
            )
            self.pipe.to(self.device)
            
            # ============ OPTIMIZATION 3: Memory Efficient Attention ============
            try:
                # Try xFormers (fastest)
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("✅ xFormers memory efficient attention enabled")
            except Exception:
                try:
                    # Fallback to SDPA (PyTorch 2.0+)
                    from diffusers.models.attention_processor import AttnProcessor2_0
                    self.pipe.unet.set_attn_processor(AttnProcessor2_0())
                    logger.info("✅ SDPA attention processor enabled")
                except Exception:
                    logger.info("⚠️ Using default attention")
            
            # ============ OPTIMIZATION 4: VAE Optimizations ============
            try:
                # Enable VAE slicing for memory efficiency
                self.pipe.enable_vae_slicing()
                logger.info("✅ VAE slicing enabled")
            except Exception:
                pass
            
            try:
                # Enable VAE tiling for large images
                self.pipe.enable_vae_tiling()
                logger.info("✅ VAE tiling enabled")
            except Exception:
                pass
            
            # ============ OPTIMIZATION 5: torch.compile (PyTorch 2.0+) ============
            if hasattr(torch, 'compile') and os.getenv("ENABLE_COMPILE", "false").lower() == "true":
                try:
                    self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
                    logger.info("✅ torch.compile enabled for UNet")
                except Exception as e:
                    logger.warning(f"⚠️ torch.compile failed: {e}")
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded and optimized in {load_time:.2f}s")
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
            
            logger.info(f"📝 Prompt {idx}: {input_text[:80]}...")
            inputs.append(input_text)
        return inputs

    def inference(self, inputs):
        """Generate images with optimized settings"""
        logger.info(f"🎨 Generating {len(inputs)} image(s) with OPTIMIZED settings...")
        
        start_time = time.time()
        
        try:
            # ============ OPTIMIZATION 6: Reduced Steps ============
            # Z-Image-Turbo is a distilled model, works well with fewer steps
            NUM_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", "25"))  # Reduced from 50
            GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "7.0"))
            
            logger.info(f"⚙️ Steps: {NUM_STEPS}, Guidance: {GUIDANCE_SCALE}")
            
            # ============ OPTIMIZATION 7: Disable gradient computation ============
            with torch.inference_mode():
                images = self.pipe(
                    inputs,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE
                ).images
            
            inference_time = time.time() - start_time
            logger.info(f"✅ Generated {len(images)} image(s) in {inference_time:.2f}s")
            logger.info(f"⏱️ Speed: {inference_time/len(images):.2f}s per image")
            
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
