import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from models import Autoencoder, CombinedLoss, HuberLoss, DTWLoss, FrequencyLoss, CombinedDTWLoss
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
    'sequence_length': 128,

    'loss_function': CombinedDTWLoss(alpha=0.2), # Reduced alpha to emphasize L1 loss
    'normalise': False,
    'random_sampling': True,
    # Training parameters
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 30,
    'train_split': 0.8,
    
    # Model parameters
    'hidden_size': 64,
    'latent_dim': 12,
    'num_layers': 4,  # Number of hidden layers in encoder/decoder
    
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
    """Reconstruct a sample and save the reconstruction with the specified loss function"""
    with torch.no_grad():
        reconstruction = model(sample)
        
        # Calculate loss using the specified criterion
        loss = criterion(reconstruction, sample).item()
        
        # Move tensors to CPU and convert to numpy for plotting
        input_seq = sample[0].cpu().numpy()
        output_seq = reconstruction[0].cpu().numpy()
        
        # Get the name of the loss function
        loss_name = criterion.__class__.__name__
        
        # Plot with loss metric
        plt.figure(figsize=(12, 4))
        plt.plot(input_seq, 'b-', label='Input', alpha=0.7)
        plt.plot(output_seq, 'r-', label='Reconstruction', alpha=0.7)
        plt.title(f'{sample_name} - {loss_name} - Epoch {epoch+1}\n' +
                 f'Loss: {loss:.6f}')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Value')

        plt.legend()
        plt.grid(True)
        plt.savefig(f'reconstructions/reconstruction_epoch_{epoch+1}_{sample_name}.png')
        plt.close()
        
        return loss

def visualize_model_architecture(params):
    """Create ASCII visualization of model architecture"""
    sequence_length = params['sequence_length']
    n_features = len(params['selected_columns'])
    hidden_size = params['hidden_size']
    latent_dim = params['latent_dim']
    num_layers = params['num_layers']
    input_dim = sequence_length * n_features
    
    # Calculate layer dimensions for visualization
    def calculate_layer_dims(start_dim, end_dim, num_layers):
        if num_layers == 1:
            return [start_dim, end_dim]
        ratio = (end_dim / start_dim) ** (1 / (num_layers))
        dims = [int(start_dim * (ratio ** i)) for i in range(num_layers + 1)]
        return dims
    
    # Get encoder and decoder dimensions
    encoder_dims = calculate_layer_dims(input_dim, latent_dim, num_layers)
    decoder_dims = calculate_layer_dims(latent_dim, input_dim, num_layers)
    
    print("\nModel Architecture:")
    print("=" * 80)
    
    # Build the architecture string
    architecture = f"""
        Input Signal ({sequence_length} timesteps × {n_features} features = {input_dim})
            ↓
        [Encoder]"""
    
    # Add encoder layers
    for i in range(num_layers):
        architecture += f"""
            ↓
        Hidden Layer E{i+1} ({encoder_dims[i+1]} units)
            ↓ ReLU"""
    
    architecture += f"""
        Latent Space ({latent_dim} units) ← Compressed Representation
            ↓
        [Decoder]"""
    
    # Add decoder layers
    for i in range(num_layers):
        architecture += f"""
            ↓
        Hidden Layer D{i+1} ({decoder_dims[i+1]} units)
            ↓ ReLU"""
    
    architecture += f"""
        Output Signal ({sequence_length} timesteps × {n_features} features = {input_dim})
            """
    
    print(architecture)
    print("=" * 80)
            
    # Calculate compression ratio
    compression_ratio = input_dim / latent_dim
    print(f"Compression ratio: {compression_ratio:.1f}:1 ({input_dim} → {latent_dim})")

def calculate_reconstruction_errors(model, dataloader, device, criterion):
    """Calculate reconstruction errors for a dataset using the specified loss function"""
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstruction = model(batch)
            # Calculate loss using the specified criterion
            # L1Loss returns a scalar per batch, so we don't need to take mean across dimensions
            errors = criterion(reconstruction, batch)
            reconstruction_errors.extend([errors.item()])
    
    return np.array(reconstruction_errors)

def evaluate_anomaly_detection(model, normal_loader, fault_loader, threshold, device):
    """Evaluate anomaly detection performance"""
    # Calculate reconstruction errors for normal and fault data
    normal_errors = calculate_reconstruction_errors(model, normal_loader, device, HYPERPARAMETERS['loss_function'])
    fault_errors = calculate_reconstruction_errors(model, fault_loader, device, HYPERPARAMETERS['loss_function'])
    
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

def plot_error_distributions(normal_errors, fault_errors, threshold, epoch, criterion, save_dir='validation'):
    """Plot distribution of reconstruction errors"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get the name of the loss function
    loss_name = criterion.__class__.__name__
    
    plt.figure(figsize=(10, 6))
    plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(fault_errors, bins=50, alpha=0.5, label='Fault', density=True)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title(f'Reconstruction Error Distribution ({loss_name}) - Epoch {epoch+1}')
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
        normalise=HYPERPARAMETERS['normalise'],
        random_sampling=HYPERPARAMETERS['random_sampling'],
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
        normalise=HYPERPARAMETERS['normalise'],
        random_sampling=HYPERPARAMETERS['random_sampling'],
        # n_fft=(HYPERPARAMETERS['sequence_length']*2)-1,
        # return_magnitude=True
    )[0]  # Take only the first loader since we don't need to split fault data
    
    # Initialize model, optimizer, and loss function
    model = Autoencoder(
        sequence_length=params['sequence_length'],
        n_features=len(params['selected_columns']),
        hidden_dim=params['hidden_size'],
        latent_dim=params['latent_dim'],
        num_layers=params['num_layers']
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = HYPERPARAMETERS['loss_function']
    
    # Training loop
    train_losses = []
    
    
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
            test_batch = next(iter(test_loader))
            random_idx = random.randint(0, test_batch.size(0) - 1)
            test_sample = test_batch[random_idx].unsqueeze(0).to(device)  # Add batch dimension
            
            fault_test_batch = next(iter(fault_loader))
            random_idx = random.randint(0, fault_test_batch.size(0) - 1)
            fault_test_sample = fault_test_batch[random_idx].unsqueeze(0).to(device)  # Add batch dimension

            reconstruct(model, test_sample,epoch, criterion, 'normal')
            reconstruct(model, fault_test_sample,epoch, criterion, 'underhang_35g_bearing_fault')

            # Calculate reconstruction errors on validation set
            val_errors = calculate_reconstruction_errors(model, test_loader, device, HYPERPARAMETERS['loss_function'])
            
            # Set threshold based on normal data distribution
            threshold = np.percentile(val_errors, params['reconstruction_error_percentile'])
            
            # Evaluate anomaly detection performance
            metrics = evaluate_anomaly_detection(model, test_loader, fault_loader, threshold, device)
            
            # Plot error distributions
            plot_error_distributions(
                metrics['normal_errors'],
                metrics['fault_errors'],
                threshold,
                epoch,
                HYPERPARAMETERS['loss_function']
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
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': params,
        'final_loss': train_losses[-1],
        'anomaly_threshold': best_threshold,
        'best_metrics': best_metrics
    }, 'autoencoder_checkpoint.pth')
    
    # Also save just the model state dict for easier conversion
    torch.save(model.state_dict(), 'autoencoder_model.pth')
    
    print(f"Model saved to autoencoder_model.pth")

if __name__ == "__main__":
    visualize_model_architecture(HYPERPARAMETERS)
    train_model()

