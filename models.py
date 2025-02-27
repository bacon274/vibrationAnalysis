import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, sequence_length=100, n_features=1, hidden_dim=64, latent_dim=16):
        """
        Args:
            sequence_length (int): Length of the input sequence
            n_features (int): Number of features (columns) in input
            hidden_dim (int): Size of hidden layers
            latent_dim (int): Size of the latent space
        """
        super(Autoencoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.input_dim = sequence_length * n_features
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, self.input_dim)
        )

    def forward(self, x):
        # Flatten the input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten sequence and features
        
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        # Reshape back to sequence format
        decoded = decoded.view(batch_size, self.sequence_length, self.n_features)
        
        return decoded

    def encode(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.encoder(x)


class MLP(nn.Module,):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2))

    def forward(self, x):
        x = self.mlp(x)
        return x
    

