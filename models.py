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
    



class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        abs_error = torch.abs(y_pred - y_true)
        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        return torch.mean(0.5 * quadratic.pow(2) + self.delta * linear)


class DTWLoss(nn.Module):
    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        loss = 0
        for i in range(batch_size):
            D = torch.cdist(y_pred[i].unsqueeze(-1), y_true[i].unsqueeze(-1))
            # Compute cumulative distance matrix
            for j in range(1, D.shape[0]):
                for k in range(1, D.shape[1]):
                    D[j, k] += torch.min(torch.tensor([D[j-1, k], D[j, k-1], D[j-1, k-1]]))
            loss += D[-1, -1]
        return loss / batch_size


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        # MSE for point-wise accuracy
        mse_loss = self.mse(y_pred, y_true)
        
        # Pattern loss using gradients
        y_pred_grad = y_pred[:, 1:] - y_pred[:, :-1]
        y_true_grad = y_true[:, 1:] - y_true[:, :-1]
        pattern_loss = self.mse(y_pred_grad, y_true_grad)
        
        return (1 - self.alpha) * mse_loss + self.alpha * pattern_loss


class FrequencyLoss(nn.Module):
    def forward(self, y_pred, y_true):
        # Convert to frequency domain
        y_pred_fft = torch.fft.rfft(y_pred, dim=1)
        y_true_fft = torch.fft.rfft(y_true, dim=1)
        
        # Compute loss in frequency domain
        return torch.mean(torch.abs(y_pred_fft - y_true_fft))