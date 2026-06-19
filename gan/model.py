"""
GAN Model (mirroring VAE architecture)
────────────────────────────────────────
Generator  = VAE Decoder   (ConvTranspose + Tanh)
Discriminator = VAE Encoder (Conv + scalar output)

Parameter counts are nearly identical to the VAE’s encoder/decoder.
"""

import torch
import torch.nn as nn
from typing import Optional


class Generator(nn.Module):
    """
    Mirrors VAE Decoder exactly, except final activation = Tanh.

    Architecture (default channels [256, 128, 64, 32]):
        Linear(latent_dim → 256*4*4)
        Reshape → (B, 256, 4, 4)
        Bottleneck (3x3 conv)
        ConvTranspose2d(256→128, 4, stride=2, pad=1) → 8×8
        ConvTranspose2d(128→64,  4, stride=2, pad=1) → 16×16
        ConvTranspose2d(64→32,   4, stride=2, pad=1) → 32×32
        ConvTranspose2d(32→1,    4, stride=2, pad=1) → 64×64
        Tanh()
    """

    def __init__(self, config) -> None:
        super().__init__()
        # config has: latent_dim, gen_channels (e.g., [256,128,64,32]),
        #             image_size, in_channels
        channels = config.gen_channels
        spatial = config.image_size // (2 ** len(channels))   # e.g., 64//16 = 4
        self.spatial = spatial
        self.first_channels = channels[0]
        flat_dim = channels[0] * spatial * spatial

        self.fc = nn.Sequential(
            nn.Linear(config.latent_dim, flat_dim),
            nn.BatchNorm1d(flat_dim),
            nn.ReLU(inplace=True),
        )
        # Bottleneck: extra conv to refine the initial feature map (same as VAE Decoder)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Transposed convolutions
        layers = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.ConvTranspose2d(
                    channels[i], channels[i+1], 4, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
            ])
        # Final layer: output image with Tanh (values in [-1, 1])
        layers.extend([
            nn.ConvTranspose2d(
                channels[-1], config.in_channels, 4, stride=2, padding=1, bias=False
            ),
            nn.Tanh(),
        ])
        self.deconv = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Same Kaiming init as the VAE Decoder."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self.first_channels, self.spatial, self.spatial)
        h = self.bottleneck(h)
        return self.deconv(h)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Discriminator(nn.Module):
    """
    Mirrors VAE Encoder, but outputs a single scalar (no Sigmoid).

    Architecture (default channels [32, 64, 128, 256]):
        Conv2d(1→32,  4, stride=2, pad=1) → 32×32
        LayerNorm + LeakyReLU
        Conv2d(32→64, 4, stride=2, pad=1) → 16×16
        LayerNorm + LeakyReLU
        Conv2d(64→128,4, stride=2, pad=1) → 8×8
        LayerNorm + LeakyReLU
        Conv2d(128→256,4,stride=2, pad=1) → 4×4
        LayerNorm + LeakyReLU
        Flatten → 256*4*4 = 4096
        Linear(4096 → 1)
    Note: LayerNorm is used instead of BatchNorm to avoid WGAN-GP issues.
    """

    def __init__(self, config) -> None:
        super().__init__()
        channels = [config.in_channels] + config.disc_channels  # [1,32,64,128,256]

        layers = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.Conv2d(
                    channels[i], channels[i+1], 4, stride=2, padding=1, bias=False
                ),
                nn.LayerNorm([
                    channels[i+1],
                    config.image_size // (2**(i+1)),
                    config.image_size // (2**(i+1))
                ]),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        self.conv = nn.Sequential(*layers)

        # Compute flatten size (same as VAE Encoder)
        spatial = config.image_size // (2 ** len(config.disc_channels))   # 64//16 = 4
        flat_dim = config.disc_channels[-1] * spatial * spatial           # 256*4*4 = 4096
        self.fc = nn.Linear(flat_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Same Kaiming init as VAE Encoder, but using 'fan_in' mode for LeakyReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)