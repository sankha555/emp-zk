import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse

class SimpleNN(nn.Module):
    """Simple fully connected network."""
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(SimpleNN, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        self.input_size = input_size
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)

def pgd_attack(model, x, y, epsilon, alpha, num_steps):
    """
    PGD adversarial attack.
    
    Args:
        model: neural network
        x: input batch
        y: true labels
        epsilon: maximum perturbation
        alpha: step size
        num_steps: number of PGD steps
    """
    x_adv = x.clone().detach()
    
    for _ in range(num_steps):
        x_adv.requires_grad = True
        
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, y)
        
        grad = torch.autograd.grad(loss, x_adv)[0]
        
        # Update adversarial example
        x_adv = x_adv.detach() + alpha * grad.sign()
        
        # Project back to epsilon ball
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1).detach()
    
    return x_adv

def train_epoch(model, train_loader, optimizer, device, epsilon, alpha, num_steps):
    """Train for one epoch with adversarial training."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Generate adversarial examples
        data_adv = pgd_attack(model, data, target, epsilon, alpha, num_steps)
        
        optimizer.zero_grad()
        output = model(data_adv)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch {batch_idx+1}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, device, epsilon=None, alpha=None, num_steps=None):
    """Test the model on clean or adversarial examples."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad() if epsilon is None else torch.enable_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if epsilon is not None:
                # Test on adversarial examples
                data = pgd_attack(model, data, target, epsilon, alpha, num_steps)
            
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy

def get_dataset(dataset_name, batch_size):
    """Load MNIST or CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    if dataset_name == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        input_size = 28 * 28
        num_classes = 10
    elif dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        input_size = 32 * 32 * 3
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, input_size, num_classes

def main():
    parser = argparse.ArgumentParser(description='Adversarial Training')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--hidden', type=int, nargs='+', default=[100, 100], 
                        help='Hidden layer sizes (e.g., --hidden 100 100 100)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epsilon', type=float, default=0.1, 
                        help='Adversarial perturbation radius')
    parser.add_argument('--alpha', type=float, default=0.01, 
                        help='PGD step size')
    parser.add_argument('--pgd-steps', type=int, default=10, 
                        help='Number of PGD steps')
    parser.add_argument('--output', type=str, default='model.onnx')
    parser.add_argument('--no-cuda', action='store_true')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading {args.dataset.upper()} dataset...")
    train_loader, test_loader, input_size, num_classes = get_dataset(
        args.dataset, args.batch_size
    )
    
    # Create model
    print(f"\nCreating model with architecture: {input_size} -> {' -> '.join(map(str, args.hidden))} -> {num_classes}")
    model = SimpleNN(input_size, args.hidden, num_classes).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training
    print(f"\nStarting adversarial training (epsilon={args.epsilon}, alpha={args.alpha}, steps={args.pgd_steps})")
    print("="*80)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, 
            args.epsilon, args.alpha, args.pgd_steps
        )
        
        clean_acc = test(model, test_loader, device)
        adv_acc = test(model, test_loader, device, args.epsilon, args.alpha, args.pgd_steps)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Clean Acc: {clean_acc:.2f}%")
        print(f"  Test Adversarial Acc: {adv_acc:.2f}%")
    
    # Save as ONNX
    print(f"\nSaving model to {args.output}")
    model.eval()
    dummy_input = torch.randn(1, input_size).to(device)
    torch.onnx.export(
        model, 
        dummy_input, 
        args.output,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Training complete!")

if __name__ == '__main__':
    main()