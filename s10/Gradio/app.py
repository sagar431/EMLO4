import gradio as gr
import torch
import timm
from PIL import Image

class ImageClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create model and move to appropriate device
        self.model = timm.create_model('mambaout_base.in1k', pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get model specific transforms
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

        # Load ImageNet labels
        import requests
        url = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
        self.labels = requests.get(url).text.strip().split('\n')

    @torch.no_grad()
    def predict(self, image):
        """ORIGINAL: Process ONE image at a time"""
        if image is None:
            return None

        # Preprocess image
        img = Image.fromarray(image).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # Shape: [1, 3, 224, 224]

        # Get prediction - ONE image
        output = self.model(img_tensor)  # Shape: [1, 1000]
        probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Shape: [1000]

        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        return {
            self.labels[idx.item()]: float(prob)
            for prob, idx in zip(top5_prob, top5_catid)
        }

    @torch.no_grad()
    def predict_batch(self, image_list, progress=gr.Progress(track_tqdm=True)):
        """BATCHED: Process MULTIPLE images together"""
        # Step 1: Handle different input formats
        if isinstance(image_list, tuple) and len(image_list) == 1:
            image_list = [image_list[0]]  # Convert single image tuple to list

        if not image_list or image_list[0] is None:
            return [[{"none": 1.0}]]  # Return empty result

        progress(0.1, desc="Starting preprocessing...")

        # Step 2: Preprocess ALL images in the batch
        tensors = []
        for image in image_list:
            if image is None:
                continue
            img = Image.fromarray(image).convert('RGB')  # Convert to PIL Image
            tensor = self.transform(img)  # Apply transforms (resize, normalize, etc.)
            tensors.append(tensor)  # Collect all tensors

        if not tensors:
            return [[{"none": 1.0}]]

        progress(0.4, desc="Batching tensors...")

        # Step 3: STACK all tensors into ONE big batch
        # Before: [tensor1], [tensor2], [tensor3], [tensor4]
        # After:  [tensor1, tensor2, tensor3, tensor4] as one tensor
        batch = torch.stack(tensors).to(self.device)  # Shape: [batch_size, 3, 224, 224]

        progress(0.6, desc="Running inference...")

        # Step 4: SINGLE model call for ENTIRE batch!
        # Before: 4 separate model calls
        # After:  1 model call processing 4 images together
        outputs = self.model(batch)  # Shape: [batch_size, 1000]
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Shape: [batch_size, 1000]

        progress(0.8, desc="Processing results...")

        # Step 5: Process results for EACH image in the batch
        batch_results = []
        for probs in probabilities:  # Loop through each image's probabilities
            top5_prob, top5_catid = torch.topk(probs, 5)
            result = {
                self.labels[idx.item()]: float(prob)
                for prob, idx in zip(top5_prob, top5_catid)
            }
            batch_results.append(result)

        progress(1.0, desc="Done!")

        # Step 6: Return results in Gradio's expected format
        # Gradio expects: [[result1], [result2], [result3], [result4]]
        return [batch_results]

# Create classifier instance
classifier = ImageClassifier()

# Choose which function to use:
USE_BATCHING = True  # Set to False to use original single-image processing

if USE_BATCHING:
    # BATCHED VERSION: Processes multiple images together
    demo = gr.Interface(
        fn=classifier.predict_batch,  # Use batch function
        inputs=gr.Image(),
        outputs=gr.Label(num_top_classes=5),
        title="Image Classification with BATCHING",
        description="Upload images - they get processed in batches for better performance!",
        examples=[
            ["golden_retriever.png"],
            ["tabby_cat.png"],
            ["sports_car.png"]
        ],
        batch=True,              # ENABLE batching!
        max_batch_size=4,        # Max 4 images per batch
        batch_duration=0.5       # Wait 0.5 seconds to collect requests
    )
else:
    # ORIGINAL VERSION: Processes one image at a time
    demo = gr.Interface(
        fn=classifier.predict,   # Use single-image function
        inputs=gr.Image(),
        outputs=gr.Label(num_top_classes=5),
        title="Basic Image Classification (Single Image)",
        description="Upload an image to classify it using the mambaout_base.in1k model",
        examples=[
            ["golden_retriever.png"],
            ["tabby_cat.png"],
            ["sports_car.png"]
        ]
        # batch=False (default)
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True  # Enable verbose error reporting
    ) 
