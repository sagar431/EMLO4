import os
import json
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import Net

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy

def main():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transform),
        batch_size=1000, shuffle=True)

    model = Net().to(device)
    
    checkpoint_path = '/mnist/model/mnist_cnn.pt'
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found!")
        return
    
    model.load_state_dict(torch.load(checkpoint_path))
    
    test_loss, accuracy = test(model, device, test_loader)
    
    results = {
        "Test loss": test_loss,
        "Accuracy": accuracy
    }
    
    os.makedirs('/mnist/model', exist_ok=True)
    with open('/mnist/model/eval_results.json', 'w') as f:
        json.dump(results, f)
    
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
