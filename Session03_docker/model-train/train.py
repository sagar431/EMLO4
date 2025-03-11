import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import datasets, transforms

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Net

def train_epoch(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train_hogwild(rank, model, device, dataset, dataloader_kwargs, epochs=1, lr=0.01, momentum=0.5, seed=1):
    torch.manual_seed(seed + rank)
    train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    for epoch in range(1, epochs + 1):
        train_epoch(model, device, train_loader, optimizer, epoch)

def main():
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Set parameters
    batch_size = 64
    test_batch_size = 1000
    epochs = 1
    lr = 0.01
    momentum = 0.5
    seed = 1
    num_processes = 2
    
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
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    
    kwargs = {'batch_size': batch_size, 'shuffle': True}
    if use_cuda:
        kwargs.update({'num_workers': 1, 'pin_memory': True})
    
    # Check if checkpoint exists
    checkpoint_path = os.path.join(model_dir, 'mnist_cnn.pt')
    model = Net().to(device)
    
    if os.path.exists(checkpoint_path):
        print("Checkpoint found, resuming training...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found, starting training from scratch...")
    
    # Share model memory for Hogwild training
    model.share_memory()
    
    # Start Hogwild training processes
    torch.manual_seed(seed)
    mp.set_start_method('spawn', force=True)
    
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train_hogwild, args=(rank, model, device, train_dataset, kwargs, epochs, lr, momentum, seed))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Save the model
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

if __name__ == '__main__':
    main() 