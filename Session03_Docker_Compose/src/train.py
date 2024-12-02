import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from model import Net

def train(rank, args, model, device, dataloader, optimizer):
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()

def main():
    # Training settings
    batch_size = 64
    test_batch_size = 1000
    epochs = 1
    lr = 0.01
    momentum = 0.5
    seed = 1
    num_processes = 2
    cuda = torch.cuda.is_available()

    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Net().to(device)
    model.share_memory()  # Required for Hogwild training

    checkpoint_path = '/mnist/model/mnist_cnn.pt'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        model.load_state_dict(torch.load(checkpoint_path))
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(rank, None, model, device, train_loader, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Saving checkpoint...")
    torch.save(model.state_dict(), checkpoint_path)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
