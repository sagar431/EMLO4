import torch
import timm
from PIL import Image
import io
import litserve as ls
import base64

class ImageClassifierAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize the model and necessary components"""
        self.device = device
        # Create model and move to appropriate device
        self.model = timm.create_model('mambaout_base.in1k', pretrained=True)
        self.model = self.model.to(device)
        self.model.eval()

        # Get model specific transforms
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        # Load ImageNet labels
        import requests
        url = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
        self.labels = requests.get(url).text.strip().split('\n')

    def decode_request(self, request):
        """Convert base64 encoded image to tensor"""
        image_bytes = request.get("image")
        if not image_bytes:
            raise ValueError("No image data provided")
        return image_bytes
    
    def batch(self, inputs):
        """Process and batch multiple inputs"""
        batched_tensors = []
        for image_bytes in inputs:
            # Decode base64 string to bytes
            img_bytes = base64.b64decode(image_bytes)
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(img_bytes))
            # Transform image to tensor
            tensor = self.transforms(image)
            batched_tensors.append(tensor)
            
        # Stack all tensors into a batch
        return torch.stack(batched_tensors).to(self.device)

    @torch.no_grad()
    def predict(self, x):
        """Run inference on the input batch"""
        outputs = self.model(x)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities
    
    def unbatch(self, output):
        """Split batch output into individual predictions"""
        return [output[i] for i in range(len(output))]

    def encode_response(self, output):
        """Convert model output to API response"""
        # Get top 5 predictions
        probs, indices = torch.topk(output, k=5)
        
        return {
            "predictions": [
                {
                    "label": self.labels[idx.item()],
                    "probability": prob.item()
                }
                for prob, idx in zip(probs, indices)
            ]
        }

if __name__ == "__main__":
    api = ImageClassifierAPI()
    # Configure server with batching
    server = ls.LitServer(
        api,
        accelerator="gpu",
        max_batch_size=64,  # Adjust based on your GPU memory and requirements
        batch_timeout=0.01,  # Timeout in seconds to wait for forming batches
    )
    server.run(port=8000)