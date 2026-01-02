import os
import sys
import random
import torch
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    print("Warning: matplotlib not found. Will save raw tensor data instead.")
    HAS_MATPLOTLIB = False
from torchvision import datasets, transforms

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Net

def main():
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    data_dir = '/opt/mount/data'
    model_dir = '/opt/mount/model'
    results_dir = '/opt/mount/results'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Data loading and transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    # Check if checkpoint exists
    checkpoint_path = os.path.join(model_dir, 'mnist_cnn.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)
    
    # Load model
    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Select 5 random images from the test dataset
    indices = random.sample(range(len(test_dataset)), 5)
    
    # Run inference on the selected images
    with torch.no_grad():
        for i, idx in enumerate(indices):
            data, target = test_dataset[idx]
            data = data.to(device)
            
            # Get prediction
            output = model(data.unsqueeze(0))
            pred = output.argmax(dim=1, keepdim=True).item()
            
            # Save the image with the predicted label as the filename
            img = data.cpu().squeeze().numpy()
            
            if HAS_MATPLOTLIB:
                # Create a figure and save it
                plt.figure(figsize=(3, 3))
                plt.imshow(img, cmap='gray')
                plt.title(f"Predicted: {pred}, Actual: {target}")
                plt.axis('off')
                plt.tight_layout()
                
                # Save the image with the predicted number as the filename
                plt.savefig(f"{results_dir}/{pred}_{i}.png")
                plt.close()
            else:
                # Save as raw numpy array if matplotlib is not available
                import numpy as np
                np.save(f"{results_dir}/{pred}_{i}.npy", img)
            
            print(f"Saved image {i+1}/5 with prediction {pred} (actual: {target})")
    
    print(f"Inference completed. Results saved to {results_dir}")

if __name__ == '__main__':
    main()
