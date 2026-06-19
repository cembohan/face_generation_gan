"""
VAE Training Configuration
──────────────────────────
Central configuration for all hyperparameters, paths, and training settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VAEConfig:
    """Full configuration for VAE training pipeline."""

    # ── Paths ──────────────────────────────────────────────────────────────
    project_root: Path = Path(".")
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("runs")
    output_dir: Path = Path("outputs")
    data_cache_dir: Path = Path("data")

    # ── Data ───────────────────────────────────────────────────────────────
    image_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True

    # ── Model ──────────────────────────────────────────────────────────────
    in_channels: int = 1
    latent_dim: int = 128
    encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    decoder_channels: list[int] = field(default_factory=lambda: [256, 128, 64, 32])

    # ── Training ───────────────────────────────────────────────────────────
    epochs: int = 300
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_max_norm: float = 0.5

    # ── KL Annealing ──────────────────────────────────────────────────────
    kl_warmup_epochs: int = 50
    kl_weight_max: float = 0.5
    kl_anneal_strategy: str = "cyclical"  # "linear" | "cyclical"
    kl_cyclical_period: int = 30  # only for cyclical

    # ── Reconstruction Loss ───────────────────────────────────────────────
    recon_loss: str = "mse"  # "mse" | "bce"

    # ── LR Scheduler ─────────────────────────────────────────────────────
    scheduler: str = "cosine"  # "cosine" | "plateau" | "none"
    scheduler_patience: int = 15  # for plateau
    scheduler_min_lr: float = 1e-6

    # ── Mixed Precision ──────────────────────────────────────────────────
    use_amp: bool = True

    # ── Checkpointing ────────────────────────────────────────────────────
    save_every_n_epochs: int = 25
    keep_top_k: int = 3

    # ── Logging ──────────────────────────────────────────────────────────
    log_images_every_n_epochs: int = 5
    num_reconstruction_samples: int = 8
    num_random_samples: int = 16

    # ── Reproducibility ──────────────────────────────────────────────────
    seed: int = 42

    # ── Early Stopping ───────────────────────────────────────────────────
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4

    # ── Generation ───────────────────────────────────────────────────────
    gen_grid_rows: int = 5
    gen_grid_cols: int = 5
    gen_temperature: float = 1.0
    gen_checkpoint: Optional[str] = None  # path to checkpoint for generation

    def __post_init__(self) -> None:
        """Ensure all directories exist."""
        for d in [self.checkpoint_dir, self.log_dir, self.output_dir, self.data_cache_dir]:
            d.mkdir(parents=True, exist_ok=True)
