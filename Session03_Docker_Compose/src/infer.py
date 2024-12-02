import os
import torch
from torchvision import datasets, transforms
from model import Net
import random
from PIL import Image

def main():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load test dataset
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Load model
    model = Net().to(device)
    checkpoint_path = '/mnist/model/mnist_cnn.pt'
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found!")
        return
    
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    # Create results directory
    os.makedirs('/mnist/results', exist_ok=True)
    
    # Select 5 random images
    indices = random.sample(range(len(test_dataset)), 5)
    
    with torch.no_grad():
        for idx in indices:
            data, _ = test_dataset[idx]
            output = model(data.unsqueeze(0).to(device))
            pred = output.argmax(dim=1, keepdim=True).item()
            
            # Convert tensor to PIL Image and save
            img = data.squeeze().numpy()
            img = ((img * 0.3081) + 0.1307) * 255
            img = Image.fromarray(img).convert('L')
            img.save(f'/mnist/results/predicted_{pred}.png')
