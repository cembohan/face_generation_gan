"""
GAN Training Configuration
──────────────────────────
Central configuration for all hyperparameters, paths, and training settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GANConfig:
    """Full configuration for GAN training pipeline."""

    # Resolve project root dynamically so it is always absolute and points to the parent directory of gan/
    project_root: Path = Path(__file__).resolve().parent.parent
    checkpoint_dir: Path = Path(__file__).resolve().parent / "checkpoints_gan"
    log_dir: Path = Path(__file__).resolve().parent / "runs_gan"
    output_dir: Path = Path(__file__).resolve().parent / "outputs_gan"
    data_cache_dir: Path = Path(__file__).resolve().parent.parent / "data"

    # Data
    image_size: int = 64
    num_workers: int = 8  # Set to 0 to avoid DataLoader multiprocessing hang on Windows
    pin_memory: bool = True

    # Model (same dimensions as VAE)
    in_channels: int = 1
    latent_dim: int = 128
    gen_channels: list[int] = field(default_factory=lambda: [256, 128, 64, 32])   # decoder style
    disc_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])  # encoder style

    # Training
    epochs: int = 300
    batch_size: int = 256
    g_lr: float = 1e-4
    d_lr: float = 1e-4
    beta1: float = 0.5          # Adam β1 (common for GANs)
    beta2: float = 0.9
    n_critic: int = 2           # discriminator updates per generator update
    gp_weight: float = 10.0     # gradient penalty coefficient
    grad_clip_max_norm: float = 0.0  # 0 = disabled (not needed for WGAN-GP)

    # Scheduler (optional, often omitted for GANs)
    scheduler: str = "none"     # "cosine", "plateau", "none"
    scheduler_min_lr: float = 1e-6
    scheduler_patience: int = 15

    # Mixed Precision
    use_amp: bool = True

    # Checkpointing
    save_every_n_epochs: int = 1
    keep_top_k: int = 3

    # Logging
    log_images_every_n_epochs: int = 5
    num_random_samples: int = 16

    # Reproducibility
    seed: int = 42

    # Early stopping
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4

    # Generation
    gen_grid_rows: int = 5
    gen_grid_cols: int = 5
    gen_temperature: float = 1.0   # not used for GAN, can ignore
    gen_checkpoint: Optional[str] = None

    def __post_init__(self):
        for d in [self.checkpoint_dir, self.log_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
