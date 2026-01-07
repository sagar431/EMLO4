"""
Optimized Cat-Dog Classifier Server using LitServe

Optimizations:
1. Batching - Process multiple images together (max_batch_size=64)
2. Workers - Multiple workers per GPU (workers_per_device=4)
3. Half Precision - Use bfloat16 for faster inference
4. Parallel Decoding - Use ThreadPoolExecutor for image preprocessing
"""

import torch
import timm
from PIL import Image
import io
import litserve as ls
import base64
from concurrent.futures import ThreadPoolExecutor
import os

# ImageNet class indices for cats and dogs
CAT_CLASSES = list(range(281, 286))  # 281-285
DOG_CLASSES = list(range(151, 269))  # 151-268

# Precision setting
PRECISION = torch.bfloat16


class CatDogClassifierAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize the model and necessary components"""
        self.device = device
        
        # Create model with half precision
        self.model = timm.create_model('resnet18', pretrained=True)
        self.model = self.model.to(device).to(PRECISION)
        self.model.eval()
        
        # Compile model for faster inference (PyTorch 2.0+)
        # self.model = torch.compile(self.model)

        # Get model specific transforms
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        # Load ImageNet labels
        import requests
        url = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
        self.imagenet_labels = requests.get(url).text.strip().split('\n')
        
        # Thread pool for parallel image decoding
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

    def decode_request(self, request):
        """Extract base64 image - actual decoding happens in batch()"""
        image_bytes = request.get("image")
        if not image_bytes:
            raise ValueError("No image data provided")
        return image_bytes
    
    def _process_single_image(self, image_bytes):
        """Process a single image (used in parallel)"""
        img_bytes = base64.b64decode(image_bytes)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tensor = self.transforms(image)
        return tensor
    
    def batch(self, inputs):
        """Process and batch multiple inputs using parallel processing"""
        # Process images in parallel using thread pool
        tensors = list(self.executor.map(self._process_single_image, inputs))
        
        # Stack into batch and convert to half precision
        batched = torch.stack(tensors).to(self.device).to(PRECISION)
        return batched

    @torch.no_grad()
    def predict(self, x):
        """Run inference on the batch"""
        outputs = self.model(x)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Process each item in batch
        results = []
        for i in range(len(probabilities)):
            probs = probabilities[i]
            cat_prob = probs[CAT_CLASSES].sum().item()
            dog_prob = probs[DOG_CLASSES].sum().item()
            top_prob, top_idx = probs.max(dim=0)
            
            results.append({
                "cat_prob": cat_prob,
                "dog_prob": dog_prob,
                "top_imagenet_label": self.imagenet_labels[top_idx.item()],
                "top_imagenet_prob": top_prob.item()
            })
        
        return results
    
    def unbatch(self, outputs):
        """Split batch results into individual predictions"""
        return outputs

    def encode_response(self, output):
        """Convert model output to API response"""
        cat_prob = output["cat_prob"]
        dog_prob = output["dog_prob"]
        
        total = cat_prob + dog_prob
        if total > 0:
            cat_pct = cat_prob / total
            dog_pct = dog_prob / total
        else:
            cat_pct = dog_pct = 0.5
        
        prediction = "cat" if cat_prob > dog_prob else "dog"
        confidence = max(cat_pct, dog_pct)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "cat_probability": cat_pct,
            "dog_probability": dog_pct,
            "details": {
                "raw_cat_prob": output["cat_prob"],
                "raw_dog_prob": output["dog_prob"],
                "top_imagenet_label": output["top_imagenet_label"],
                "top_imagenet_confidence": output["top_imagenet_prob"]
            }
        }


if __name__ == "__main__":
    api = CatDogClassifierAPI()
    
    # Optimized server configuration
    server = ls.LitServer(
        api,
        accelerator="gpu",
        max_batch_size=64,          # Batch up to 64 requests
        batch_timeout=0.05,         # Wait 50ms to form batches
        workers_per_device=4,       # 4 workers per GPU
    )
    server.run(port=8000)
