import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

# ==================== Dataset Loading Functions ====================

# def load_adult_dataset(filepath='adult.csv'):
#     """
#     Load and preprocess Adult Income dataset
#     Target: >50K or <=50K
#     """
#     # If file doesn't exist, download it
#     if not os.path.exists(filepath):
#         print("Downloading Adult dataset...")
#         url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
#         df = pd.read_csv(url, header=None, na_values=' ?', skipinitialspace=True)
        
#         # Column names
#         df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
#                      'marital-status', 'occupation', 'relationship', 'race', 'sex',
#                      'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
#         df.to_csv(filepath, index=False)
#     else:
#         df = pd.read_csv(filepath, na_values=' ?', skipinitialspace=True)
    
#     # Drop rows with missing values
#     df = df.dropna()
    
#     # Encode target
#     df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)
    
#     # Label encode categorical variables (convert to numerical)
#     categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
#                        'relationship', 'race', 'sex', 'native-country']
    
#     le = LabelEncoder()
#     for col in categorical_cols:
#         df[col] = le.fit_transform(df[col].astype(str))
    
#     # Separate features and target
#     X = df.drop('income', axis=1).values.astype(np.float32)
#     y = df['income'].values
    
#     return X, y, df.drop('income', axis=1).columns.tolist()\
    
def load_adult_dataset(filepath='adult_samples.csv'):
    df = pd.read_csv(filepath, na_values=' ?', skipinitialspace=True)
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Encode target
    print(df.columns)
    # df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)
    print("hello")
    
    # Label encode categorical variables (convert to numerical)
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country']
    
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Separate features and target
    X = df.drop('income', axis=1).values.astype(np.float32)
    y = df['income'].values
    
    return X, y, df.drop('income', axis=1).columns.tolist()
    

def load_default_credit_dataset(filepath='default_credit.csv'):
    """
    Load and preprocess Default of Credit Card Clients dataset
    Target: default payment (1=yes, 0=no)
    """
    if not os.path.exists(filepath):
        print("Downloading Default Credit dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
        df = pd.read_excel(url, header=1)
        df.to_csv(filepath, index=False)
    else:
        df = pd.read_csv(filepath)
    
    # Remove ID column if present
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    # Target is usually the last column or named 'default payment next month'
    target_col = 'default payment next month' if 'default payment next month' in df.columns else df.columns[-1]
    
    X = df.drop(target_col, axis=1).values.astype(np.float32)
    y = df[target_col].values
    
    return X, y, df.drop(target_col, axis=1).columns.tolist()

def load_german_credit_dataset(filepath='german_credit.csv'):
    """
    Load and preprocess German Credit dataset
    Target: credit risk (1=good, 2=bad -> convert to 0=good, 1=bad)
    """
    if not os.path.exists(filepath):
        print("Downloading German Credit dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        df = pd.read_csv(url, sep=' ', header=None)
        
        # Add column names
        cols = [f'attr{i}' for i in range(1, 21)] + ['credit_risk']
        df.columns = cols
        df.to_csv(filepath, index=False)
    else:
        df = pd.read_csv(filepath)
    
    # Convert target: 1=good (0), 2=bad (1)
    df['credit_risk'] = df['credit_risk'].apply(lambda x: 0 if x == 1 else 1)
    
    # Label encode any categorical columns
    for col in df.columns:
        if col != 'credit_risk' and df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    X = df.drop('credit_risk', axis=1).values.astype(np.float32)
    y = df['credit_risk'].values
    
    return X, y, df.drop('credit_risk', axis=1).columns.tolist()

# ==================== Neural Network Architecture ====================

class CustomNN(nn.Module):
    """
    Customizable Neural Network with specified architecture
    """
    def __init__(self, input_size, hidden_layers, output_size=2, activation='relu'):
        super(CustomNN, self).__init__()
        
        # layers = []
        # prev_size = input_size
        
        # # Build hidden layers
        # for hidden_size in hidden_layers:
        #     layers.append(nn.Linear(prev_size, hidden_size))
            
        #     if activation == 'relu':
        #         layers.append(nn.ReLU())
        #     elif activation == 'tanh':
        #         layers.append(nn.Tanh())
        #     elif activation == 'sigmoid':
        #         layers.append(nn.Sigmoid())
            
        #     prev_size = hidden_size
        
        # # Output layer with 2 neurons
        # layers.append(nn.Linear(prev_size, output_size))
        # # No activation here - will use softmax in loss function
        
        
        layers = [
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
            nn.ReLU()
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# ==================== Training Function ====================

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=50, device='cpu', patience=10):
    """
    Train the neural network with early stopping
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                model.load_state_dict(best_model_state)
                break
    
    return train_losses, val_losses, train_accs, val_accs

# ==================== Evaluation Function ====================

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate the model on test set
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'\nTest Accuracy: {accuracy*100:.2f}%')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_predictions))
    print('\nConfusion Matrix:')
    print(confusion_matrix(all_labels, all_predictions))
    
    return accuracy

# ==================== Main Training Pipeline ====================

def save_dataset_txt(dataset, out_path):
    with open(out_path, "w") as f:
        for x, y in dataset:
            # label first
            f.write(str(int(y.item())))
            # then attributes
            for v in x:
                f.write(f" {v.item()}")
            f.write("\n")


def train_on_dataset(dataset_name, X, y, hidden_layers, num_epochs=100, 
                     batch_size=64, learning_rate=0.001, test_size=0.2):
    """
    Complete training pipeline for a dataset
    """
    print(f"\n{'='*80}")
    print(f"Training on {dataset_name} dataset")
    print(f"Architecture: {X.shape[1]} -> {' -> '.join(map(str, hidden_layers))} -> 1")
    print(f"{'='*80}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Further split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    save_dataset_txt(test_dataset, f"test/ai/data/inputs/{dataset_name}_test_1.txt")
    
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomNN(X.shape[1], hidden_layers, output_size=2)
    
    # Loss and optimizer (CrossEntropyLoss for 2-class classification)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.2)
    
    # Train model
    print("Training started...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=num_epochs, device=device, patience=15
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy = evaluate_model(model, test_loader, device=device)
    
    # Save model
    model_path = f"{dataset_name}_model.onnx"
    model.eval()
    dummy_input = torch.randn(1, X_test.shape[1], device = device)
    print(dummy_input.shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=11,          # important
        do_constant_folding=False, # important for ZK / fixed-point
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,         # disable dynamic axes
        dynamo = False
    )
    
    # torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{dataset_name} - Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{dataset_name} - Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_training_curves.png', dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {dataset_name}_training_curves.png")
    plt.close()
    
    return model, test_accuracy

# ==================== Main Execution ====================

def main():
    # Define architecture (can be customized)
    hidden_layers = [128, 64, 32]  # Example: 3 hidden layers
    
    # Training parameters
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.01
    
    results = {}
    
    import sys
    dataset = sys.argv[1]
    
    if dataset == "adult":
        # Train on Adult dataset
        try:
            print("\n" + "="*80)
            print("ADULT INCOME DATASET")
            print("="*80)
            X, y, feature_names = load_adult_dataset()
            print(f"Dataset shape: {X.shape}, Target distribution: {np.bincount(y)}")
            model, acc = train_on_dataset(
                'adult', X, y, hidden_layers, num_epochs, batch_size, learning_rate
            )
            results['adult'] = acc
        except Exception as e:
            print(f"Error with Adult dataset: {e}")
    
    
    if dataset == "credit":
        # Train on Default Credit dataset
        try:
            print("\n" + "="*80)
            print("DEFAULT CREDIT CARD DATASET")
            print("="*80)
            X, y, feature_names = load_default_credit_dataset()
            print(f"Dataset shape: {X.shape}, Target distribution: {np.bincount(y)}")
            model, acc = train_on_dataset(
                'default_credit', X, y, hidden_layers, num_epochs, batch_size, learning_rate
            )
            results['default_credit'] = acc
        except Exception as e:
            print(f"Error with Default Credit dataset: {e}")
   
    
    if dataset == "german":
        # Train on German Credit dataset
        try:
            print("\n" + "="*80)
            print("GERMAN CREDIT DATASET")
            print("="*80)
            X, y, feature_names = load_german_credit_dataset()
            print(f"Dataset shape: {X.shape}, Target distribution: {np.bincount(y)}")
            model, acc = train_on_dataset(
                'german_credit', X, y, hidden_layers, num_epochs, batch_size, learning_rate
            )
            results['german_credit'] = acc
        except Exception as e:
            print(f"Error with German Credit dataset: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    for dataset, accuracy in results.items():
        print(f"{dataset}: Test Accuracy = {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
    
