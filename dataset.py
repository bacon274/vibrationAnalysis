import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq



class VibrationDataset(Dataset):
    def __init__(self, csv_path, selected_columns=None, sequence_length=100):
        """
        Args:
            csv_path (str): Path to the CSV file
            selected_columns (list): List of column names to use. If None, uses all columns
            sequence_length (int): Length of the sequence to return
        """
        # Read the CSV file
        self.df = pd.read_csv(csv_path)
        
        # Use selected columns if provided, otherwise use all columns
        if selected_columns is not None:
            self.df = self.df[selected_columns]
        
        # Convert to numpy array for faster processing
        self.data = self.df.values
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Get sequence of data
        sequence = self.data[idx:idx + self.sequence_length]
        return torch.FloatTensor(sequence)

class FrequencyDomainDataset(VibrationDataset):
    def __init__(self, csv_path, selected_columns=None, sequence_length=100, 
                 n_fft=None, return_magnitude=True):
        """
        Args:
            csv_path (str): Path to the CSV file
            selected_columns (list): List of column names to use
            sequence_length (int): Length of the sequence to return
            n_fft (int): Number of FFT points. If None, uses sequence_length
            return_magnitude (bool): If True, returns magnitude spectrum, else returns complex FFT
        """
        super().__init__(csv_path, selected_columns, sequence_length)
        self.n_fft = n_fft or sequence_length
        self.return_magnitude = return_magnitude
    
    def __getitem__(self, idx):
        # Get time domain sequence
        sequence = self.data[idx:idx + self.sequence_length]
        
        # Compute FFT
        freq_domain = fft(sequence, n=self.n_fft, axis=0)
        
        # Optional: return only first half of spectrum (due to symmetry)
        freq_domain = freq_domain[:self.n_fft//2 + 1]
        
        if self.return_magnitude:
            # Convert to magnitude spectrum
            freq_domain = np.abs(freq_domain)
        # else:
        #     # Stack real and imaginary parts
        #     freq_domain = np.stack((freq_domain.real, freq_domain.imag), axis=-1)
            
        return torch.FloatTensor(freq_domain)

def create_data_loaders(csv_path, selected_columns=None, sequence_length=100, 
                       batch_size=32, train_split=0.8, shuffle=True,
                       domain='time', **freq_kwargs):
    """
    Creates train and test dataloaders for either time or frequency domain data.
    
    Args:
        csv_path (str): Path to the CSV file
        selected_columns (list): List of column names to use
        sequence_length (int): Length of each sequence
        batch_size (int): Size of each batch
        train_split (float): Proportion of data to use for training
        shuffle (bool): Whether to shuffle the data
        domain (str): 'time' or 'freq' to specify domain
        **freq_kwargs: Additional arguments for FrequencyDomainDataset
    
    Returns:
        train_loader, test_loader: DataLoader objects for training and testing
    """
    # Create appropriate dataset
    if domain == 'time':
        dataset = VibrationDataset(csv_path, selected_columns, sequence_length)
    elif domain == 'freq':
        dataset = FrequencyDomainDataset(csv_path, selected_columns, 
                                       sequence_length, **freq_kwargs)
    else:
        raise ValueError("domain must be 'time' or 'freq'")
    
    # Calculate lengths for train/test split
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    
    # Split dataset
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

# Example usage:
if __name__ == "__main__":
    # Example of how to use the dataset and dataloader
    csv_path = 'data/combined/normal_500Hz.csv'
    
    # Select specific columns
    selected_columns = [
        'underhang_bearing_axial',
        # 'underhang_bearing_radial',
        # 'underhang_bearing_tangential'
    ]
    
    # Create train and test loaders
    train_loader, test_loader = create_data_loaders(
        csv_path=csv_path,
        selected_columns=selected_columns,
        sequence_length=100,
        batch_size=32,
        domain='freq',
        n_fft=1024,
        return_magnitude=True
    )
    
    # Print sample batch
    for batch in train_loader:
        print("Batch shape:", batch.shape)
        print("Sample sequence:\n", batch[0])
        break 