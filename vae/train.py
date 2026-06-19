"""
VAE Training Script (Corrected)
--------------------------------
Fixed:
  - Cyclical KL annealing now follows a proper triangular wave
  - Early stopping and checkpoint ranking use the fixed-weight validation loss
  - BCE reconstruction loss works correctly with [0,1] outputs (Sigmoid in decoder)
  - Checkpoint manager simplified, best model saved externally for clarity
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from config import VAEConfig
from dataset import create_dataloaders   # your dataset module
from model import VAE                    # your model module


# =============================================================================
# Loss Functions
# =============================================================================

def reconstruction_loss(x_recon: torch.Tensor, x: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
    """
    Compute reconstruction loss.
      - 'mse':  mean squared error (sum / batch)
      - 'bce':  binary cross-entropy for probabilities (x_recon in [0,1])
    """
    if loss_type == "bce":
        # Numerically stable BCE for probabilities
        eps = 1e-7
        x_recon = x_recon.clamp(eps, 1.0 - eps)
        bce = -(x * torch.log(x_recon) + (1.0 - x) * torch.log(1.0 - x_recon))
        return bce.sum() / x.size(0)
    else:
        return F.mse_loss(x_recon, x, reduction="sum") / x.size(0)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence (summed over latent dims, averaged over batch)."""
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def compute_kl_weight(epoch: int, config: VAEConfig) -> float:
    """
    Compute KL annealing weight.
      - 'linear': ramp from 0 to kl_weight_max over kl_warmup_epochs, then constant.
      - 'cyclical': triangular wave with period kl_cyclical_period, peaks at kl_weight_max.
    """
    if config.kl_anneal_strategy == "cyclical":
        cycle = epoch % config.kl_cyclical_period
        half = config.kl_cyclical_period / 2.0
        if cycle < half:
            return (cycle / half) * config.kl_weight_max
        else:
            return (2.0 - cycle / half) * config.kl_weight_max
    else:  # linear warmup
        if epoch >= config.kl_warmup_epochs:
            return config.kl_weight_max
        return config.kl_weight_max * (epoch / config.kl_warmup_epochs)


# =============================================================================
# Training Utilities
# =============================================================================

def seed_everything(seed: int) -> None:
    """Set all seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CheckpointManager:
    """Keep top‑k checkpoints by validation loss, removing the rest."""

    def __init__(self, checkpoint_dir: Path, keep_top_k: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.best_checkpoints: list[tuple[float, Path]] = []

    def save(
        self,
        model: VAE,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        epoch: int,
        val_loss: float,
        config: VAEConfig,
    ) -> Path:
        """Save checkpoint and prune old ones. Returns path to saved file."""
        ckpt_path = self.checkpoint_dir / f"vae_epoch{epoch:04d}_loss{val_loss:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "latent_dim": config.latent_dim,
                    "encoder_channels": config.encoder_channels,
                    "decoder_channels": config.decoder_channels,
                    "image_size": config.image_size,
                    "in_channels": config.in_channels,
                },
            },
            ckpt_path,
        )

        self.best_checkpoints.append((val_loss, ckpt_path))
        self.best_checkpoints.sort(key=lambda x: x[0])

        while len(self.best_checkpoints) > self.keep_top_k:
            _, rm_path = self.best_checkpoints.pop()
            if rm_path.exists():
                rm_path.unlink()

        return ckpt_path


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# =============================================================================
# TensorBoard Logging
# =============================================================================

@torch.no_grad()
def log_images(
    writer: SummaryWriter,
    model: VAE,
    val_batch: torch.Tensor,
    device: torch.device,
    epoch: int,
    config: VAEConfig,
) -> None:
    """Log reconstruction comparisons and random samples to TensorBoard."""
    model.eval()

    # Reconstructions (interleaved originals and reconstructions)
    n = min(config.num_reconstruction_samples, val_batch.size(0))
    x = val_batch[:n].to(device)
    x_recon, _, _ = model(x)
    comparison = torch.zeros(2 * n, config.in_channels, config.image_size, config.image_size, device=device)
    comparison[0::2] = x
    comparison[1::2] = x_recon
    grid = make_grid(comparison, nrow=n, normalize=False, padding=2, pad_value=0.5)
    writer.add_image("Reconstructions/originals_vs_reconstructed", grid, epoch)

    # Random samples from prior
    samples = model.sample(config.num_random_samples, device, temperature=1.0)
    grid = make_grid(samples, nrow=4, normalize=False, padding=2, pad_value=0.5)
    writer.add_image("Samples/random_from_prior", grid, epoch)


@torch.no_grad()
def log_latent_stats(
    writer: SummaryWriter,
    model: VAE,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> None:
    """Log histograms of mu and logvar over a few validation batches."""
    model.eval()
    all_mu, all_logvar = [], []
    max_batches = 5  # avoid memory issues
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x = batch.to(device)
        mu, logvar = model.encode(x)
        all_mu.append(mu.cpu())
        all_logvar.append(logvar.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)

    writer.add_histogram("Latent/mu", all_mu, epoch)
    writer.add_histogram("Latent/logvar", all_logvar, epoch)
    writer.add_scalar("Latent/mu_mean", all_mu.mean().item(), epoch)
    writer.add_scalar("Latent/mu_std", all_mu.std().item(), epoch)
    writer.add_scalar("Latent/logvar_mean", all_logvar.mean().item(), epoch)


# =============================================================================
# Train / Eval Loops
# =============================================================================

def train_one_epoch(
    model: VAE,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    kl_weight: float,
    config: VAEConfig,
) -> dict:
    model.train()
    total_loss = total_recon = total_kl = 0.0
    n_batches = 0

    for batch in loader:
        x = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=config.use_amp):
            x_recon, mu, logvar = model(x)
            recon = reconstruction_loss(x_recon, x, config.recon_loss)
            kl = kl_divergence(mu, logvar)
            loss = recon + kl_weight * kl

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_max_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: VAE,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    kl_weight: float,
    config: VAEConfig,
) -> dict:
    model.eval()
    total_loss = total_recon = total_kl = total_fixed = 0.0
    n_batches = 0

    for batch in loader:
        x = batch.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=config.use_amp):
            x_recon, mu, logvar = model(x)
            recon = reconstruction_loss(x_recon, x, config.recon_loss)
            kl = kl_divergence(mu, logvar)
            loss = recon + kl_weight * kl

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        total_fixed += (recon + config.kl_weight_max * kl).item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "loss_fixed": total_fixed / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: VAEConfig) -> None:
    # ── Setup ─────────────────────────────────────────────────────────
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] CUDA not available, falling back to CPU (training will be slow).")
        config.use_amp = False

    print(f"\n{'='*60}")
    print(f"  VAE Training Pipeline")
    print(f"{'='*60}")
    print(f"  Device:        {device}")
    print(f"  Image size:    {config.image_size}")
    print(f"  Latent dim:    {config.latent_dim}")
    print(f"  Batch size:    {config.batch_size}")
    print(f"  Epochs:        {config.epochs}")
    print(f"  LR:            {config.learning_rate}")
    print(f"  AMP:           {config.use_amp}")
    print(f"  KL annealing:  {config.kl_anneal_strategy} (warmup={config.kl_warmup_epochs})")
    print(f"  Recon loss:    {config.recon_loss}")
    print(f"  Scheduler:     {config.scheduler}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(config)
    fixed_val_batch = next(iter(val_loader))

    # ── Model ────────────────────────────────────────────────────────
    model = VAE(config).to(device)
    print(f"\n[MODEL] Trainable parameters: {model.get_num_params():,}")

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── Scheduler ────────────────────────────────────────────────────
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.scheduler_min_lr
        )
    elif config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
        )
    else:
        scheduler = None

    scaler = GradScaler(enabled=config.use_amp)

    # ── Logging / Checkpointing / Early Stopping ─────────────────────
    run_name = f"vae_z{config.latent_dim}_bs{config.batch_size}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=config.log_dir / run_name)
    ckpt_mgr = CheckpointManager(config.checkpoint_dir, config.keep_top_k)
    early_stop = EarlyStopping(config.early_stopping_patience, config.early_stopping_min_delta)

    # Log hyperparameters
    hparams = {k: str(v) if not isinstance(v, (int, float, bool)) else v
               for k, v in vars(config).items()}
    writer.add_text("Hyperparameters", json.dumps(hparams, indent=2, default=str), 0)

    # ── Training Loop ────────────────────────────────────────────────
    best_val_loss_fixed = float("inf")
    pbar = tqdm(range(1, config.epochs + 1), desc="Training", unit="epoch")

    for epoch in pbar:
        kl_weight = compute_kl_weight(epoch, config)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, kl_weight, config
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, kl_weight, config)

        # ── LR Scheduling ─────────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss_fixed"])
            else:
                scheduler.step()

        # ── TensorBoard Scalars ───────────────────────────────────
        writer.add_scalar("Loss_Total/Train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss_Total/Validation", val_metrics["loss"], epoch)
        writer.add_scalar("Loss_Reconstruction/Train", train_metrics["recon_loss"], epoch)
        writer.add_scalar("Loss_Reconstruction/Validation", val_metrics["recon_loss"], epoch)
        writer.add_scalar("Loss_KL/Train", train_metrics["kl_loss"], epoch)
        writer.add_scalar("Loss_KL/Validation", val_metrics["kl_loss"], epoch)
        writer.add_scalar("Loss_Total/Val_FixedWeight", val_metrics["loss_fixed"], epoch)
        writer.add_scalar("Hyperparams/kl_weight", kl_weight, epoch)
        writer.add_scalar("Hyperparams/learning_rate", current_lr, epoch)

        # ── TensorBoard Images / Latent Stats ──────────────────────
        if epoch == 1 or epoch % config.log_images_every_n_epochs == 0:
            log_images(writer, model, fixed_val_batch, device, epoch, config)
            log_latent_stats(writer, model, val_loader, device, epoch)

        # ── Checkpointing ──────────────────────────────────────────
        # Use loss_fixed for model selection (always full KL weight)
        is_best = val_metrics["loss_fixed"] < best_val_loss_fixed
        if is_best:
            best_val_loss_fixed = val_metrics["loss_fixed"]

        if epoch % config.save_every_n_epochs == 0 or is_best:
            ckpt_path = ckpt_mgr.save(
                model, optimizer, scaler, epoch, val_metrics["loss_fixed"], config
            )
            if is_best:
                best_path = config.checkpoint_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss_fixed": val_metrics["loss_fixed"],
                        "config": {
                            "latent_dim": config.latent_dim,
                            "encoder_channels": config.encoder_channels,
                            "decoder_channels": config.decoder_channels,
                            "image_size": config.image_size,
                            "in_channels": config.in_channels,
                        },
                    },
                    best_path,
                )
                tqdm.write(f"  * New best model: val_loss_fixed = {best_val_loss_fixed:.6f}")

        # ── Progress bar update ────────────────────────────────────
        pbar.set_postfix({
            "t_loss": f"{train_metrics['loss']:.3f}",
            "v_fixed": f"{val_metrics['loss_fixed']:.3f}",
            "kl_w": f"{kl_weight:.2f}",
            "lr": f"{current_lr:.1e}",
        })

        # ── Early Stopping ─────────────────────────────────────────
        if early_stop.step(val_metrics["loss_fixed"]):
            tqdm.write(f"\n[EARLY STOP] No improvement for {config.early_stopping_patience} epochs. Stopping.")
            break

    # ── Cleanup ────────────────────────────────────────────────────
    writer.close()
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val_loss_fixed: {best_val_loss_fixed:.6f}")
    print(f"  Checkpoints saved to: {config.checkpoint_dir.resolve()}")
    print(f"  TensorBoard:          {config.log_dir.resolve() / run_name}")
    print(f"\n  Launch TensorBoard:  tensorboard --logdir={config.log_dir}")
    print(f"{'='*60}\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> VAEConfig:
    config = VAEConfig()

    parser = argparse.ArgumentParser(
        description="Train a Variational Autoencoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--lr", type=float, default=config.learning_rate)
    parser.add_argument("--latent-dim", type=int, default=config.latent_dim)
    parser.add_argument("--kl-warmup", type=int, default=config.kl_warmup_epochs)
    parser.add_argument("--kl-strategy", choices=["linear", "cyclical"], default=config.kl_anneal_strategy)
    parser.add_argument("--recon-loss", choices=["mse", "bce"], default=config.recon_loss)
    parser.add_argument("--scheduler", choices=["cosine", "plateau", "none"], default=config.scheduler)
    parser.add_argument("--no-amp", action="store_true", default=not config.use_amp)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    args = parser.parse_args()

    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.latent_dim = args.latent_dim
    config.kl_warmup_epochs = args.kl_warmup
    config.kl_anneal_strategy = args.kl_strategy
    config.recon_loss = args.recon_loss
    config.scheduler = args.scheduler
    config.use_amp = not args.no_amp
    config.seed = args.seed
    config.num_workers = args.num_workers
    return config


if __name__ == "__main__":
    config = parse_args()
    train(config)