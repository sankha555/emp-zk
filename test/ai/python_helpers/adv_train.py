import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import onnx
import os

def fraction(x):
    if '/' in x:
        num, den = x.split('/')
        return float(num) / float(den)
    return float(x)


def get_model_name(dataset, hidden_sizes, epsilon, training_method='pgd'):
    num_hidden = len(hidden_sizes)
    suffix = '_diffai' if training_method == 'diffai' else ''
    return f"{dataset}_relu_{num_hidden}_{hidden_sizes[0]}_e{epsilon:.4f}{suffix}"

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
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
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
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
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
    elif dataset_name == 'cifar':
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, input_size, num_classes

# ============================================================================
# DiffAI Implementation
# ============================================================================

class IntervalBound:
    """Represents interval bounds [lower, upper] for abstract interpretation."""
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

def interval_affine(bounds, weight, bias):
    """Apply affine transformation to interval bounds."""
    center = (bounds.lower + bounds.upper) / 2
    radius = (bounds.upper - bounds.lower) / 2
    
    center_out = torch.matmul(center, weight.t()) + bias
    radius_out = torch.matmul(radius, torch.abs(weight.t()))
    
    return IntervalBound(center_out - radius_out, center_out + radius_out)

def interval_relu(bounds):
    """Apply ReLU to interval bounds."""
    lower = torch.clamp(bounds.lower, min=0)
    upper = torch.clamp(bounds.upper, min=0)
    return IntervalBound(lower, upper)

def abstract_forward(model, x_center, epsilon):
    """
    Forward pass through network with abstract interpretation.
    Returns interval bounds on the output.
    """
    x_lower = torch.clamp(x_center - epsilon, 0, 1)
    x_upper = torch.clamp(x_center + epsilon, 0, 1)
    bounds = IntervalBound(x_lower.view(x_lower.size(0), -1), 
                          x_upper.view(x_upper.size(0), -1))
    
    for layer in model.network:
        if isinstance(layer, nn.Linear):
            bounds = interval_affine(bounds, layer.weight, layer.bias)
        elif isinstance(layer, nn.ReLU):
            bounds = interval_relu(bounds)
    
    return bounds

def diffai_loss(model, x, y, epsilon, kappa=1.0):
    """
    DiffAI loss: encourages verified robustness.
    
    For correct class c and wrong class i:
    loss = max(0, upper_bound(logit_i) - lower_bound(logit_c) + kappa)
    """
    bounds = abstract_forward(model, x, epsilon)
    
    # Get bounds for correct class
    y_onehot = torch.zeros_like(bounds.lower).scatter_(1, y.unsqueeze(1), 1)
    correct_lower = (bounds.lower * y_onehot).sum(dim=1)
    
    # Get maximum upper bound for incorrect classes
    incorrect_mask = 1 - y_onehot
    incorrect_upper = bounds.upper * incorrect_mask + (-1e10) * y_onehot
    max_incorrect_upper = incorrect_upper.max(dim=1)[0]
    
    # DiffAI loss: penalize when incorrect class upper > correct class lower
    loss = torch.clamp(max_incorrect_upper - correct_lower + kappa, min=0)
    
    return loss.mean()

def train_epoch_diffai(model, train_loader, optimizer, device, epsilon, kappa, lambda_robust=0.5):
    """Train for one epoch with DiffAI."""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_diffai_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Standard cross-entropy on clean examples
        output = model(data)
        ce_loss = nn.CrossEntropyLoss()(output, target)
        
        # DiffAI loss for robustness
        diffai_loss_val = diffai_loss(model, data, target, epsilon, kappa)
        
        # Combined loss with weighting
        loss = ce_loss + lambda_robust * diffai_loss_val
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_diffai_loss += diffai_loss_val.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch {batch_idx+1}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}, DiffAI: {diffai_loss_val.item():.4f}), '
                  f'Acc: {100.*correct/total:.2f}%')
    
    avg_ce = total_ce_loss / len(train_loader)
    avg_diffai = total_diffai_loss / len(train_loader)
    print(f"  Avg CE: {avg_ce:.4f}, Avg DiffAI: {avg_diffai:.4f}")
    
    return total_loss / len(train_loader), 100. * correct / total

# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Adversarial Training')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar'])
    parser.add_argument('--hidden', type=int, nargs='+', default=[100, 100], 
                        help='Hidden layer sizes (e.g., --hidden 100 100 100)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--epsilon', type=fraction, default=0.1, 
                        help='Adversarial perturbation radius')
    parser.add_argument('--alpha', type=fraction, default=0.01, 
                        help='PGD step size')
    parser.add_argument('--pgd-steps', type=int, default=10, 
                        help='Number of PGD steps')
    parser.add_argument('--training-method', type=str, default='pgd',
                        choices=['pgd', 'diffai'],
                        help='Training method: pgd or diffai')
    parser.add_argument('--kappa', type=float, default=1.0,
                        help='Margin for DiffAI loss')
    parser.add_argument('--lambda-robust', type=float, default=0.5,
                        help='Weight for DiffAI loss (0=only CE, 1=equal weight)')
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
    
    model_name = get_model_name(args.dataset, args.hidden, args.epsilon, args.training_method)
    
    # Optimizer
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    
    # Training
    if args.training_method == 'pgd':
        print(f"\nStarting PGD adversarial training (epsilon={args.epsilon:.4f}, alpha={args.alpha:.4f}, steps={args.pgd_steps}, lr={args.lr:.4f})")
    else:
        print(f"\nStarting DiffAI training (epsilon={args.epsilon:.4f}, kappa={args.kappa}, lambda={args.lambda_robust}, lr={args.lr:.4f})")
    print("="*80)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        current_epsilon = min(args.epsilon, args.epsilon * (epoch / (args.epochs // 2)))
        
        if args.training_method == 'pgd':
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, device, 
                current_epsilon, args.alpha, args.pgd_steps
            )
        else:
            train_loss, train_acc = train_epoch_diffai(
                model, train_loader, optimizer, device,
                current_epsilon, args.kappa, args.lambda_robust
            )
        
        clean_acc = test(model, test_loader, device)
        adv_acc = test(model, test_loader, device, current_epsilon, args.alpha, args.pgd_steps)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Clean Acc: {clean_acc:.2f}%")
        print(f"  Test Adversarial Acc: {adv_acc:.2f}%")
    
    # Save as ONNX
    args.output = f"test/eran_models/{model_name}.onnx"
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
    
    model = onnx.load(args.output)
    onnx.save(model, args.output)
    os.remove(f"{args.output}.data")
    
    print("Training complete! ")
    print("Model name: ", model_name)

if __name__ == '__main__':
    main()