import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, sequence_length=100, n_features=1, hidden_dim=64, latent_dim=16, num_layers=2):
        """
        Args:
            sequence_length (int): Length of the input sequence
            n_features (int): Number of features (columns) in input
            hidden_dim (int): Size of hidden layers
            latent_dim (int): Size of the latent space
            num_layers (int): Number of hidden layers in encoder/decoder (minimum 1)
        """
        super(Autoencoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.input_dim = sequence_length * n_features
        
        # Ensure at least one layer
        num_layers = max(1, num_layers)
        
        # Calculate dimensions for each layer
        encoder_dims = self._calculate_layer_dims(self.input_dim, latent_dim, num_layers + 1)
        decoder_dims = self._calculate_layer_dims(latent_dim, self.input_dim, num_layers + 1)
        
        # Build encoder layers dynamically
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            if i < len(encoder_dims) - 2:  # No ReLU after last layer
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder layers dynamically
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i < len(decoder_dims) - 2:  # No ReLU after last layer
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def _calculate_layer_dims(self, start_dim, end_dim, num_layers):
        """
        Calculate the dimensions for each layer using geometric progression
        Args:
            start_dim (int): Input dimension
            end_dim (int): Output dimension
            num_layers (int): Number of layers (including input and output)
        Returns:
            list: Dimensions for each layer
        """
        if num_layers == 2:
            return [start_dim, end_dim]
            
        # Calculate ratio for geometric progression
        ratio = (end_dim / start_dim) ** (1 / (num_layers - 1))
        
        # Generate layer dimensions
        dims = [int(start_dim * (ratio ** i)) for i in range(num_layers - 1)]
        dims.append(end_dim)  # Ensure exact end dimension
        
        return dims

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
            # Normalize sequences to [0,1] range for scale-invariant comparison
            y_p = y_pred[i]
            y_t = y_true[i]
            y_p = (y_p - y_p.min()) / (y_p.max() - y_p.min() + 1e-8)
            y_t = (y_t - y_t.min()) / (y_t.max() - y_t.min() + 1e-8)
            
            # Compute DTW on normalized sequences
            D = torch.cdist(y_p.unsqueeze(-1), y_t.unsqueeze(-1))
            for j in range(1, D.shape[0]):
                for k in range(1, D.shape[1]):
                    D[j, k] += torch.min(torch.tensor([D[j-1, k], D[j, k-1], D[j-1, k-1]]))
            loss += D[-1, -1]
        return loss / batch_size


class CombinedDTWLoss(nn.Module):
    def __init__(self, alpha=0.2):  # Reduced alpha to give more weight to L1 loss
        super(CombinedDTWLoss, self).__init__()
        self.alpha = alpha
        self.dtw = DTWLoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, y_pred, y_true):
        dtw_loss = self.dtw(y_pred, y_true)
        l1_loss = self.l1(y_pred, y_true)
        return self.alpha * dtw_loss + (1 - self.alpha) * l1_loss


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