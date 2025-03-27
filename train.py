import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from models import Autoencoder
import matplotlib.pyplot as plt
from dataset import *
import random
import os


# Hyperparameters and Configuration
HYPERPARAMETERS = {
    # Data parameters
    'csv_path': './data/combined/normal_330Hz.csv',
    'selected_columns': [
        'underhang_bearing_axial'
    ],
    'sequence_length': 100,
    
    # Training parameters
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 60,
    'train_split': 0.8,
    
    # Model parameters
    'hidden_size': 16,
    'latent_dim': 2,
    
    # Other settings
    'random_seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Add validation parameters
    'normal_data_path': './data/combined/normal_500Hz.csv',
    'fault_data_path': './data/combined/fault_underhang_35g_bearing_500Hz.csv',  # Add path to your fault data
    'anomaly_threshold': None,  # Will be set during training based on validation
    
    # Validation parameters
    'validation_frequency': 5,  # Validate every N epochs
    'reconstruction_error_percentile': 95,  # For threshold calculation
}

# Add input_size after dictionary definition
HYPERPARAMETERS['input_size'] = len(HYPERPARAMETERS['selected_columns'])

def plot_reconstruction(input_seq, output_seq, epoch, loss, save_dir='reconstructions', sample_name=''):
    """Plot and save a comparison of input and reconstructed sequences"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    plt.plot(input_seq, 'b-', label='Input', alpha=0.7)
    plt.plot(output_seq, 'r-', label='Reconstruction', alpha=0.7)
    plt.title(f'{sample_name} - Sequence Reconstruction - Epoch {epoch+1} (Loss: {loss:.6f})')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/reconstruction_epoch_{epoch+1}_{sample_name}.png')
    plt.close()


def reconstruct(model, sample, epoch, criterion, sample_name):
    """Reconstruct a sample and save the reconstruction"""
    with torch.no_grad():
        reconstruction = model(sample)
        # Calculate loss for this sample
        test_loss = criterion(reconstruction, sample).item()
        # Move tensors to CPU and convert to numpy for plotting
        input_seq = sample[0].cpu().numpy()  # Remove batch dimension for plotting
        output_seq = reconstruction[0].cpu().numpy()  # Remove batch dimension for plotting
        plot_reconstruction(input_seq, output_seq, epoch, test_loss, 'reconstructions', sample_name)

def visualize_model_architecture(params):
    """Create ASCII visualization of model architecture"""
    sequence_length = params['sequence_length']
    n_features = len(params['selected_columns'])
    hidden_size = params['hidden_size']
    latent_dim = params['latent_dim']
    input_dim = sequence_length * n_features
    
    print("\nModel Architecture:")
    print("=" * 80)
    print(f"""
Input Signal ({sequence_length} timesteps × {n_features} features = {input_dim})
    ↓
[Encoder]
    ↓
Hidden Layer 1 ({hidden_size * 2} units)
    ↓ ReLU
Hidden Layer 2 ({hidden_size} units)
    ↓ ReLU
Latent Space ({latent_dim} units) ← Compressed Representation
    ↓
[Decoder]
    ↓
Hidden Layer 3 ({hidden_size} units)
    ↓ ReLU
Hidden Layer 4 ({hidden_size * 2} units)
    ↓ ReLU
Output Signal ({sequence_length} timesteps × {n_features} features = {input_dim})
    """)
    print("=" * 80)
    
    # Calculate compression ratio
    compression_ratio = input_dim / latent_dim
    print(f"Compression ratio: {compression_ratio:.1f}:1 ({input_dim} → {latent_dim})")

def calculate_reconstruction_errors(model, dataloader, device):
    """Calculate reconstruction errors for a dataset"""
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstruction = model(batch)
            # Calculate MSE for each sequence in the batch
            errors = torch.mean((batch - reconstruction) ** 2, dim=(1, 2))
            reconstruction_errors.extend(errors.cpu().numpy())
    
    return np.array(reconstruction_errors)

def evaluate_anomaly_detection(model, normal_loader, fault_loader, threshold, device):
    """Evaluate anomaly detection performance"""
    # Calculate reconstruction errors for normal and fault data
    normal_errors = calculate_reconstruction_errors(model, normal_loader, device)
    fault_errors = calculate_reconstruction_errors(model, fault_loader, device)
    
    # Calculate metrics
    normal_predictions = (normal_errors > threshold).astype(int)
    fault_predictions = (fault_errors > threshold).astype(int)
    
    # Calculate false positive rate (FPR) and true positive rate (TPR)
    fpr = np.mean(normal_predictions)  # False positive rate on normal data
    tpr = np.mean(fault_predictions)   # True positive rate on fault data
    
    return {
        'threshold': threshold,
        'fpr': fpr,
        'tpr': tpr,
        'normal_mean_error': np.mean(normal_errors),
        'fault_mean_error': np.mean(fault_errors),
        'normal_errors': normal_errors,
        'fault_errors': fault_errors
    }

def plot_error_distributions(normal_errors, fault_errors, threshold, epoch, save_dir='validation'):
    """Plot distribution of reconstruction errors"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(fault_errors, bins=50, alpha=0.5, label='Fault', density=True)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title(f'Reconstruction Error Distribution - Epoch {epoch+1}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/error_distribution_epoch_{epoch+1}.png')
    plt.close()

def train_model(params=HYPERPARAMETERS):
    # Set random seed for reproducibility
    torch.manual_seed(params['random_seed'])
    
    # Check if GPU is available
    device = torch.device(params['device'])
    print(f"Using device: {device}")

    # Load and prepare data using the dataset class
    train_loader, test_loader = create_data_loaders(
        csv_path=params['normal_data_path'],
        selected_columns=params['selected_columns'],
        sequence_length=params['sequence_length'],
        batch_size=params['batch_size'],
        train_split=params['train_split'],
        domain='time',
        # n_fft=(HYPERPARAMETERS['sequence_length']*2)-1,
        # return_magnitude=True
    )
    
    # Create fault data loader
    fault_loader = create_data_loaders(
        csv_path=params['fault_data_path'],
        selected_columns=params['selected_columns'],
        sequence_length=params['sequence_length'],
        batch_size=params['batch_size'],
        train_split=1.0,  # Use all fault data for testing
        domain='time',
        # n_fft=(HYPERPARAMETERS['sequence_length']*2)-1,
        # return_magnitude=True
    )[0]  # Take only the first loader since we don't need to split fault data
    
    # Initialize model, optimizer, and loss function
    model = Autoencoder(
        sequence_length=params['sequence_length'],
        n_features=len(params['selected_columns']),
        hidden_dim=params['hidden_size'],
        latent_dim=params['latent_dim']
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    
    # Get a fixed test batch for visualization
    test_batch = next(iter(test_loader))
    test_sample = test_batch[0].unsqueeze(0).to(device)  # Add batch dimension
    fault_test_batch = next(iter(fault_loader))
    fault_test_sample = fault_test_batch[0].unsqueeze(0).to(device)  # Add batch dimension
    
    print("Starting training...")
    best_threshold = None
    best_metrics = None
    
    for epoch in range(params['num_epochs']):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            loss = criterion(output, batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        print(f'Epoch [{epoch+1}/{params["num_epochs"]}], Loss: {epoch_loss:.6f}')
        
        # Evaluate and plot a test sample
        model.eval()
        # Perform validation on fault data
        if (epoch + 1) % params['validation_frequency'] == 0:
            reconstruct(model, test_sample,epoch, criterion, 'normal')
            reconstruct(model, fault_test_sample,epoch, criterion, 'underhang_35g_bearing_fault')

            # Calculate reconstruction errors on validation set
            val_errors = calculate_reconstruction_errors(model, test_loader, device)
            
            # Set threshold based on normal data distribution
            threshold = np.percentile(val_errors, params['reconstruction_error_percentile'])
            
            # Evaluate anomaly detection performance
            metrics = evaluate_anomaly_detection(model, test_loader, fault_loader, threshold, device)
            
            # Plot error distributions
            plot_error_distributions(
                metrics['normal_errors'],
                metrics['fault_errors'],
                threshold,
                epoch
            )
            
            print(f"\nValidation Metrics (Epoch {epoch+1}):")
            print(f"Threshold: {threshold:.4f}")
            print(f"False Positive Rate: {metrics['fpr']:.4f}")
            print(f"True Positive Rate: {metrics['tpr']:.4f}")
            
            # Update best threshold if this is the best performance
            if best_metrics is None or metrics['tpr'] - metrics['fpr'] > best_metrics['tpr'] - best_metrics['fpr']:
                best_threshold = threshold
                best_metrics = metrics
    
    print("Training completed!")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, params['num_epochs'] + 1), train_losses, 'b-')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    
    # Save the final model with the best threshold
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': params,
        'final_loss': train_losses[-1],
        'anomaly_threshold': best_threshold,
        'best_metrics': best_metrics
    }, 'autoencoder_model.pth')
    
    print("\nFinal Best Metrics:")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"False Positive Rate: {best_metrics['fpr']:.4f}")
    print(f"True Positive Rate: {best_metrics['tpr']:.4f}")
    print("Model and training info saved to autoencoder_model.pth")

if __name__ == "__main__":
    visualize_model_architecture(HYPERPARAMETERS)
    train_model()
