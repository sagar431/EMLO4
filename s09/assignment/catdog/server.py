"""
Cat-Dog Classifier Server using LitServe

Uses a pretrained ResNet18 model from timm. The model is trained on ImageNet which
includes multiple cat breeds (classes 281-285) and dog breeds (classes 151-268).
We aggregate these to provide Cat vs Dog classification.
"""

import torch
import timm
from PIL import Image
import io
import litserve as ls
import base64

# ImageNet class indices for cats and dogs
# Cats: 281-285 (tabby, tiger_cat, Persian_cat, Siamese_cat, Egyptian_cat)
# Dogs: 151-268 (various dog breeds)
CAT_CLASSES = list(range(281, 286))  # 281-285
DOG_CLASSES = list(range(151, 269))  # 151-268


class CatDogClassifierAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize the model and necessary components"""
        self.device = device
        
        # Create model and move to appropriate device
        # Using resnet18 for speed, can switch to resnet50 for better accuracy
        self.model = timm.create_model('resnet18', pretrained=True)
        self.model = self.model.to(device)
        self.model.eval()

        # Get model specific transforms
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        # Load ImageNet labels for debugging
        import requests
        url = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
        self.imagenet_labels = requests.get(url).text.strip().split('\n')

    def decode_request(self, request):
        """Convert base64 encoded image to tensor"""
        image_bytes = request.get("image")
        if not image_bytes:
            raise ValueError("No image data provided")
        
        # Decode base64 string to bytes
        img_bytes = base64.b64decode(image_bytes)
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Convert to tensor and move to device
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def predict(self, x):
        """Run inference and aggregate cat/dog probabilities"""
        outputs = self.model(x)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Aggregate probabilities for cats and dogs
        cat_prob = probabilities[0, CAT_CLASSES].sum().item()
        dog_prob = probabilities[0, DOG_CLASSES].sum().item()
        
        # Get top ImageNet prediction for additional info
        top_prob, top_idx = probabilities[0].max(dim=0)
        top_label = self.imagenet_labels[top_idx.item()]
        
        return {
            "cat_prob": cat_prob,
            "dog_prob": dog_prob,
            "top_imagenet_label": top_label,
            "top_imagenet_prob": top_prob.item()
        }

    def encode_response(self, output):
        """Convert model output to API response"""
        cat_prob = output["cat_prob"]
        dog_prob = output["dog_prob"]
        
        # Normalize to get cat vs dog percentage
        total = cat_prob + dog_prob
        if total > 0:
            cat_pct = cat_prob / total
            dog_pct = dog_prob / total
        else:
            cat_pct = dog_pct = 0.5
        
        # Determine prediction
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
    # Configure server without batching (baseline)
    server = ls.LitServer(
        api,
        accelerator="gpu",
    )
    server.run(port=8000)
