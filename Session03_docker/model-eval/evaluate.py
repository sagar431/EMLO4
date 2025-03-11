import os
import sys
import json
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Net

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy

def main():
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Set parameters
    test_batch_size = 1000
    
    # Create directories
    data_dir = '/opt/mount/data'
    model_dir = '/opt/mount/model'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Data loading and transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    kwargs = {'batch_size': test_batch_size, 'shuffle': True}
    if use_cuda:
        kwargs.update({'num_workers': 1, 'pin_memory': True})
    
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)
    
    # Check if checkpoint exists
    checkpoint_path = os.path.join(model_dir, 'mnist_cnn.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)
    
    # Load model
    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Evaluate model
    test_loss, accuracy = test(model, device, test_loader)
    
    # Save results to JSON
    results = {
        "Test loss": test_loss,
        "Accuracy": accuracy
    }
    
    results_path = os.path.join(model_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    
    print(f"Evaluation results saved to {results_path}")

if __name__ == '__main__':
    main()
