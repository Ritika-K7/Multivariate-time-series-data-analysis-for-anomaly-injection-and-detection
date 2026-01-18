import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Encoder: compress to low-dimensional representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

        # Decoder: reconstruct back to original size
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        # Flatten window data into vector
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        return self.decoder(z)
