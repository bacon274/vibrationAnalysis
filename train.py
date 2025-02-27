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
    'num_epochs': 30,
    'train_split': 0.8,
    
    # Model parameters
    'hidden_size': 16,
    'latent_dim': 2,
    
    # Other settings
    'random_seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Add input_size after dictionary definition
HYPERPARAMETERS['input_size'] = len(HYPERPARAMETERS['selected_columns'])

def plot_reconstruction(input_seq, output_seq, epoch, loss, save_dir='reconstructions'):
    """Plot and save a comparison of input and reconstructed sequences"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    plt.plot(input_seq, 'b-', label='Input', alpha=0.7)
    plt.plot(output_seq, 'r-', label='Reconstruction', alpha=0.7)
    plt.title(f'Sequence Reconstruction - Epoch {epoch+1} (Loss: {loss:.6f})')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/reconstruction_epoch_{epoch+1}.png')
    plt.close()

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

def train_model(params=HYPERPARAMETERS):
    # Set random seed for reproducibility
    torch.manual_seed(params['random_seed'])
    
    # Check if GPU is available
    device = torch.device(params['device'])
    print(f"Using device: {device}")

    # Load and prepare data using the dataset class
    train_loader, test_loader = create_data_loaders(
        csv_path=params['csv_path'],
        selected_columns=params['selected_columns'],
        sequence_length=params['sequence_length'],
        batch_size=params['batch_size'],
        train_split=params['train_split']
    )
    
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
    
    print("Starting training...")
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
        with torch.no_grad():
            reconstruction = model(test_sample)
            # Calculate loss for this sample
            test_loss = criterion(reconstruction, test_sample).item()
            # Move tensors to CPU and convert to numpy for plotting
            input_seq = test_sample[0].cpu().numpy()  # Remove batch dimension for plotting
            output_seq = reconstruction[0].cpu().numpy()  # Remove batch dimension for plotting
            plot_reconstruction(input_seq, output_seq, epoch, test_loss)
    
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
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': params,
        'final_loss': train_losses[-1]
    }, 'autoencoder_model.pth')
    print("Model and training info saved to autoencoder_model.pth")

if __name__ == "__main__":
    visualize_model_architecture(HYPERPARAMETERS)
    train_model()
