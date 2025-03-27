import torch
import matplotlib.pyplot as plt
import os
from models import Autoencoder
from dataset import create_data_loaders
import numpy as np

def load_model(model_path='autoencoder_model.pth'):
    """Load the trained model and its parameters"""
    # Add numpy scalar to safe globals
    import torch.serialization
    from numpy._core.multiarray import scalar
    torch.serialization.add_safe_globals([scalar])
    
    # Load with weights_only=False since we need the full checkpoint
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Get parameters from saved model
    params = checkpoint['hyperparameters']
    
    # Initialize model with saved parameters
    model = Autoencoder(
        sequence_length=params['sequence_length'],
        n_features=len(params['selected_columns']),
        hidden_dim=params['hidden_size'],
        latent_dim=params['latent_dim']
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, params, checkpoint['anomaly_threshold']

def plot_sequence(input_seq, output_seq, index, save_dir='investigation'):
    """Plot and save a single sequence comparison"""
    plt.figure(figsize=(15, 8))
    
    # Create two subplots: one for regular scale, one for normalized
    plt.subplot(2, 1, 1)
    plt.plot(input_seq, 'b-', label='Input', alpha=0.7)
    plt.plot(output_seq, 'r-', label='Reconstruction', alpha=0.7)
    plt.title(f'Sequence {index} - Original Scale')
    plt.xlabel('Time Step')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # Normalized plot
    plt.subplot(2, 1, 2)
    input_norm = (input_seq - np.min(input_seq)) / (np.max(input_seq) - np.min(input_seq) + 1e-8)
    output_norm = (output_seq - np.min(output_seq)) / (np.max(output_seq) - np.min(output_seq) + 1e-8)
    
    plt.plot(input_norm, 'b-', label='Input (normalized)', alpha=0.7)
    plt.plot(output_norm, 'r-', label='Reconstruction (normalized)', alpha=0.7)
    plt.title('Normalized Scale')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid(True)
    
    # Calculate error metrics
    mse = np.mean((input_seq - output_seq) ** 2)
    max_error = np.max(np.abs(input_seq - output_seq))
    plt.suptitle(f'MSE: {mse:.6f}, Max Error: {max_error:.6f}')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sequence_{index}.png')
    plt.close()

def investigate_fault_data():
    # Create investigation directory
    os.makedirs('investigation', exist_ok=True)
    
    # Load model and parameters
    model, params, threshold = load_model()
    device = torch.device(params['device'])
    model = model.to(device)
    
    # Create dataloader for fault data - using time domain
    fault_loader = create_data_loaders(
        csv_path=params['fault_data_path'],
        selected_columns=params['selected_columns'],
        sequence_length=params['sequence_length'],
        batch_size=1,  # Process one sequence at a time
        train_split=1.0,
        domain='time'  # Changed to time domain
    )[0]
    
    print(f"Investigating {len(fault_loader)} sequences...")
    
    # Process each sequence
    reconstruction_errors = []
    with torch.no_grad():
        for i, batch in enumerate(fault_loader):
            # Get input and reconstruction
            input_seq = batch.to(device)
            reconstruction = model(input_seq)
            
            # Convert to numpy for plotting
            input_np = input_seq[0].cpu().numpy()  # Remove batch dimension
            recon_np = reconstruction[0].cpu().numpy()  # Remove batch dimension
            
            # Plot and save
            plot_sequence(input_np, recon_np, i)
            
            # Calculate error
            error = torch.mean((input_seq - reconstruction) ** 2).item()
            reconstruction_errors.append(error)
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} sequences...")
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_errors, bins=50)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('MSE')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('investigation/error_distribution.png')
    plt.close()
    
    # Save error statistics
    with open('investigation/statistics.txt', 'w') as f:
        f.write(f"Total sequences analyzed: {len(reconstruction_errors)}\n")
        f.write(f"Mean error: {np.mean(reconstruction_errors):.6f}\n")
        f.write(f"Median error: {np.median(reconstruction_errors):.6f}\n")
        f.write(f"Min error: {np.min(reconstruction_errors):.6f}\n")
        f.write(f"Max error: {np.max(reconstruction_errors):.6f}\n")
        f.write(f"Standard deviation: {np.std(reconstruction_errors):.6f}\n")
        f.write(f"Threshold: {threshold:.6f}\n")
        f.write(f"Sequences above threshold: {sum(np.array(reconstruction_errors) > threshold)}\n")
        f.write(f"Detection rate: {sum(np.array(reconstruction_errors) > threshold) / len(reconstruction_errors):.2%}\n")

if __name__ == "__main__":
    investigate_fault_data() 