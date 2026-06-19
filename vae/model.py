"""
Convolutional Variational Autoencoder
──────────────────────────────────────
Production-grade VAE with:
  • Symmetric encoder/decoder with BatchNorm + LeakyReLU/ReLU
  • Reparameterization trick
  • Proper weight initialization (Kaiming)
  • Latent space statistics tracking
"""

import torch
import torch.nn as nn
from typing import Tuple

from config import VAEConfig


class Encoder(nn.Module):
    """
    Convolutional encoder: (B, 1, 64, 64) → mu, logvar of shape (B, latent_dim).

    Architecture (default channels [32, 64, 128, 256]):
        Conv2d(1→32,  4, stride=2, pad=1)  → 32×32
        Conv2d(32→64, 4, stride=2, pad=1)  → 16×16
        Conv2d(64→128,4, stride=2, pad=1)  → 8×8
        Conv2d(128→256,4,stride=2, pad=1)  → 4×4
        Flatten → 256×4×4 = 4096
        FC → mu (latent_dim), FC → logvar (latent_dim)
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        channels = [config.in_channels] + config.encoder_channels

        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        self.conv = nn.Sequential(*layers)

        # After 4 downsamples: 64 → 32 → 16 → 8 → 4
        spatial = config.image_size // (2 ** len(config.encoder_channels))
        flat_dim = config.encoder_channels[-1] * spatial * spatial

        self.fc_mu = nn.Linear(flat_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, config.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """
    Convolutional decoder: (B, latent_dim) → (B, 1, 64, 64).

    Mirrors the encoder architecture with ConvTranspose2d.
    Final activation: Sigmoid → outputs in [0, 1].
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        channels = config.decoder_channels

        spatial = config.image_size // (2 ** len(channels))
        self.spatial = spatial
        self.first_channels = channels[0]
        flat_dim = channels[0] * spatial * spatial

        self.fc = nn.Sequential(
            nn.Linear(config.latent_dim, flat_dim),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.ConvTranspose2d(channels[i], channels[i + 1], 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ])
        # Final layer → single channel, sigmoid
        layers.extend([
            nn.ConvTranspose2d(channels[-1], config.in_channels, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        ])
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self.first_channels, self.spatial, self.spatial)
        return self.deconv(h)


class VAE(nn.Module):
    """
    Full VAE combining encoder, reparameterization, and decoder.

    Methods:
        encode(x)       → (mu, logvar)
        decode(z)       → x_recon
        reparameterize  → z sampled via reparameterization trick
        forward(x)      → (x_recon, mu, logvar)
        sample(n, device, temperature) → generated samples
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialization for conv/linear, constant for batchnorm."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @torch.no_grad()
    def sample(
        self, n: int, device: torch.device, temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate n samples from the prior N(0, temperature^2 * I)."""
        z = torch.randn(n, self.config.latent_dim, device=device) * temperature
        return self.decode(z)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
