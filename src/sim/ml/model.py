import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=None, latent_dim: int = 16):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
