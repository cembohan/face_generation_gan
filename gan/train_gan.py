"""
WGAN-GP Training Pipeline
─────────────────────────
Fully detailed implementation for Wasserstein GAN with Gradient Penalty.
Optimized for 64x64 grayscale CelebA training on Windows hardware.
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
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.v2 as transforms
from torchvision.datasets import CelebA
from torchvision.utils import make_grid
from tqdm import tqdm

from config import GANConfig
from model import Generator, Discriminator


# =============================================================================
# DATA LOADER SYSTEM (With Clean Label Interception)
# =============================================================================

class UnsupervisedCelebA(Dataset):
    """
    Wraps Torchvision's CelebA dataset to cleanly isolate the image tensor,
    preventing DataLoader collation errors when target attributes are omitted.
    """
    def __init__(self, celeba_dataset: CelebA) -> None:
        super().__init__()
        self.dataset = celeba_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self.dataset[idx]
        return img


def get_celeba_transforms(config: GANConfig, is_train: bool) -> transforms.Compose:
    """Transforms raw images to grayscale tensors matching the pipeline footprint."""
    transform_list = [
        transforms.CenterCrop(178),
        transforms.Resize((config.image_size, config.image_size)),
    ]

    if config.in_channels == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    transform_list.append(transforms.ToImage())
    transform_list.append(transforms.ToDtype(torch.float32, scale=True))
    
    # Scale to [-1, 1] range required for Tanh Generator outputs
    transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    if is_train:
        transform_list.insert(0, transforms.RandomHorizontalFlip(p=0.5))

    return transforms.Compose(transform_list)


def create_dataloaders(config: GANConfig) -> tuple[DataLoader, DataLoader]:
    """Generates train and validation data streaming configurations."""
    print(f"[DATA] Initializing CelebA datasets out of: {config.data_cache_dir.resolve()}")

    raw_train = CelebA(
        root=str(config.data_cache_dir),
        split="train",
        target_type="attr",
        transform=get_celeba_transforms(config, is_train=True),
        download=True,
    )

    raw_val = CelebA(
        root=str(config.data_cache_dir),
        split="valid",
        target_type="attr",
        transform=get_celeba_transforms(config, is_train=False),
        download=True,
    )

    train_set = UnsupervisedCelebA(raw_train)
    val_set = UnsupervisedCelebA(raw_val)

    # Windows-safe multi-threading fallback assignment
    workers = config.num_workers

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=workers > 0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        persistent_workers=workers > 0,
    )

    print(f"[DATA] Train dataset ready: {len(train_set)} samples ({len(train_loader)} batches)")
    print(f"[DATA] Val dataset ready:   {len(val_set)} samples ({len(val_loader)} batches)")

    return train_loader, val_loader


# =============================================================================
# WGAN-GP REGULARIZATION MATHEMATICS
# =============================================================================

def gradient_penalty(discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Enforces the 1-Lipschitz smoothness condition across the interpolation space
    to stabilize adversarial training.
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Mix real and fake distributions randomly
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = discriminator(interpolates)

    grad_outputs = torch.ones_like(d_interpolates, device=device)
    
    # Explicitly calculate tracking gradients across mixed samples
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


# =============================================================================
# INFRASTRUCTURE UTILITIES
# =============================================================================

def seed_everything(seed: int) -> None:
    """Locks hardware seeds for structural reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CheckpointManager:
    """Manages periodic checkpoint saves on disk, retaining a rolling top-k."""
    def __init__(self, checkpoint_dir: Path, keep_top_k: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.saved_paths: list[Path] = []

    def save(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        g_opt: torch.optim.Optimizer,
        d_opt: torch.optim.Optimizer,
        scaler: GradScaler,
        epoch: int,
        w_dist: float,
        config: GANConfig,
    ) -> Path:
        ckpt_path = self.checkpoint_dir / f"gan_epoch{epoch:04d}_wdist{w_dist:.4f}.pt"
        
        torch.save(
            {
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "g_optimizer_state_dict": g_opt.state_dict(),
                "d_optimizer_state_dict": d_opt.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "w_dist": w_dist,
                "config": vars(config),
            },
            ckpt_path,
        )

        self.saved_paths.append(ckpt_path)
        if len(self.saved_paths) > self.keep_top_k:
            oldest_file = self.saved_paths.pop(0)
            if oldest_file.exists():
                oldest_file.unlink()
                
        return ckpt_path


# =============================================================================
# METRIC & IMAGE LOGGING
# =============================================================================

@torch.no_grad()
def log_images(writer: SummaryWriter, generator: nn.Module, fixed_noise: torch.Tensor, device: torch.device, epoch: int) -> None:
    """Outputs visual validation grids to tracking logs."""
    generator.eval()
    samples = generator(fixed_noise.to(device))
    
    # Rescale from [-1, 1] back to [0, 1] for visual tracking outputs
    grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1), padding=2)
    writer.add_image("Generated/Samples", grid, epoch)
    generator.train()


# =============================================================================
# STEP OPERATIONS
# =============================================================================

def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    g_opt: torch.optim.Optimizer,
    d_opt: torch.optim.Optimizer,
    scaler: GradScaler,
    dataloader: DataLoader,
    device: torch.device,
    config: GANConfig,
) -> dict[str, float]:
    """Runs a single macro optimization pass across the train dataset batches."""
    generator.train()
    discriminator.train()

    d_losses, g_losses, gp_values = [], [], []

    for real_imgs in dataloader:
        real_imgs = real_imgs.to(device, non_blocking=True)
        batch_size = real_imgs.size(0)

        # ---------------------------------------------------------------------
        # 1. OPTIMIZE DISCRIMINATOR (n_critic Iterations)
        # ---------------------------------------------------------------------
        for _ in range(config.n_critic):
            d_opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=config.use_amp):
                # Process ground-truth samples
                d_real = discriminator(real_imgs)
                d_real_loss = -d_real.mean()

                # Process synthetic samples
                z = torch.randn(batch_size, config.latent_dim, device=device)
                # 1. Keep the standard real/fake calculations in AMP
                with autocast(device_type="cuda", enabled=config.use_amp):
                    d_real = discriminator(real_imgs)
                    d_real_loss = -d_real.mean()
                    
                    z = torch.randn(batch_size, config.latent_dim, device=device)
                    with torch.no_grad():
                        fake_imgs = generator(z)
                    d_fake = discriminator(fake_imgs)
                    d_fake_loss = d_fake.mean()

                # 2. Drop OUT of autocast to calculate the GP safely in float32
                gp = gradient_penalty(discriminator, real_imgs, fake_imgs, device)

                # 3. Re-enter or add them together safely
                d_loss = d_real_loss + d_fake_loss + config.gp_weight * gp

            scaler.scale(d_loss).backward()
            scaler.unscale_(d_opt)
            if config.grad_clip_max_norm > 0:
                nn.utils.clip_grad_norm_(discriminator.parameters(), config.grad_clip_max_norm)
            scaler.step(d_opt)
            scaler.update()

        d_losses.append(d_loss.item())
        gp_values.append(gp.item())

        # ---------------------------------------------------------------------
        # 2. OPTIMIZE GENERATOR (1 Iteration)
        # ---------------------------------------------------------------------
        g_opt.zero_grad(set_to_none=True)
        
        with autocast(device_type="cuda", enabled=config.use_amp):
            z = torch.randn(batch_size, config.latent_dim, device=device)
            fake_imgs = generator(z)
            d_fake = discriminator(fake_imgs)
            
            # Inverse score direction to mask synthetic data as authentic
            g_loss = -d_fake.mean()

        scaler.scale(g_loss).backward()
        scaler.unscale_(g_opt)
        if config.grad_clip_max_norm > 0:
            nn.utils.clip_grad_norm_(generator.parameters(), config.grad_clip_max_norm)
        scaler.step(g_opt)
        scaler.update()

        g_losses.append(g_loss.item())

    return {
        "d_loss": float(np.mean(d_losses)),
        "g_loss": float(np.mean(g_losses)),
        "gp": float(np.mean(gp_values)),
    }


@torch.no_grad()
def evaluate(generator: nn.Module, discriminator: nn.Module, dataloader: DataLoader, device: torch.device, config: GANConfig) -> dict[str, float]:
    """Measures empirical distance over validation splits without altering parameters."""
    generator.eval()
    discriminator.eval()

    w_distances = []
    for real_imgs in dataloader:
        real_imgs = real_imgs.to(device, non_blocking=True)
        batch_size = real_imgs.size(0)
        
        with autocast(device_type="cuda", enabled=config.use_amp):
            d_real = discriminator(real_imgs)
            z = torch.randn(batch_size, config.latent_dim, device=device)
            fake_imgs = generator(z)
            d_fake = discriminator(fake_imgs)
            
            # Empirical Wasserstein Distance: Mean(Real) - Mean(Fake)
            w_dist = d_real.mean() - d_fake.mean()
            
        w_distances.append(w_dist.item())

    return {"w_dist": float(np.mean(w_distances))}


# =============================================================================
# PIPELINE EXECUTION ENTRYPOINT
# =============================================================================

def train(config: GANConfig) -> None:
    """Coordinates parameters, loops data streams, and saves structural updates."""
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] CUDA missing! Defaulting to slow CPU calculations.")
        config.use_amp = False

    print(f"\n{'='*60}\n  WGAN-GP Training Pipeline\n{'='*60}")
    print(f"  Device:         {device}")
    print(f"  Image Size:     {config.image_size}x{config.image_size}")
    print(f"  Latent Dim:     {config.latent_dim}")
    print(f"  Batch Size:     {config.batch_size}")
    print(f"  n_critic:       {config.n_critic}")
    print(f"  GP Coefficient: {config.gp_weight}")
    print(f"  AMP Acceleration: {config.use_amp}")
    print(f"{'='*60}\n")

    train_loader, val_loader = create_dataloaders(config)
    fixed_noise = torch.randn(config.num_random_samples, config.latent_dim, device=device)
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)

    print(f"\n[MODEL] Generator Parameters:    {sum(p.numel() for p in generator.parameters()):,}")
    print(f"[MODEL] Discriminator Parameters:  {sum(p.numel() for p in discriminator.parameters()):,}")

    g_opt = torch.optim.Adam(generator.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2))

    # Optional learning rate decay schedules
    if config.scheduler == "cosine":
        g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=config.epochs, eta_min=config.scheduler_min_lr)
        d_sched = torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=config.epochs, eta_min=config.scheduler_min_lr)
    else:
        g_sched = d_sched = None

    scaler = GradScaler(enabled=config.use_amp)
    run_name = f"gan_z{config.latent_dim}_bs{config.batch_size}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=config.log_dir / run_name)
    ckpt_mgr = CheckpointManager(config.checkpoint_dir, config.keep_top_k)

    # Save meta parameter states
    hparams = {k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in vars(config).items()}
    writer.add_text("Hyperparameters", json.dumps(hparams, indent=2), 0)

    pbar = tqdm(range(1, config.epochs + 1), desc="Training", unit="epoch")
    
    for epoch in pbar:
        train_metrics = train_one_epoch(generator, discriminator, g_opt, d_opt, scaler, train_loader, device, config)
        val_metrics = evaluate(generator, discriminator, val_loader, device, config)

        if g_sched is not None:
            g_sched.step()
            d_sched.step()

        # Structural TensorBoard Tracking Updates
        writer.add_scalar("Loss/Discriminator", train_metrics["d_loss"], epoch)
        writer.add_scalar("Loss/Generator", train_metrics["g_loss"], epoch)
        writer.add_scalar("Loss/GP", train_metrics["gp"], epoch)
        writer.add_scalar("Metrics/Wasserstein_Distance", val_metrics["w_dist"], epoch)

        if epoch == 1 or epoch % config.log_images_every_n_epochs == 0:
            log_images(writer, generator, fixed_noise, device, epoch)

        # Periodic Save Pipeline
        if epoch % config.save_every_n_epochs == 0:
            ckpt_mgr.save(generator, discriminator, g_opt, d_opt, scaler, epoch, val_metrics["w_dist"], config)
            tqdm.write(f"  * Checkpoint saved at epoch {epoch} | Val W-Dist: {val_metrics['w_dist']:.4f}")

        pbar.set_postfix({
            "D_loss": f"{train_metrics['d_loss']:.2f}",
            "G_loss": f"{train_metrics['g_loss']:.2f}",
            "W_Dist": f"{val_metrics['w_dist']:.3f}",
        })

    writer.close()
    print(f"\n{'='*60}\n Training Complete! Configuration Logged Successfully.\n{'='*60}\n")


# =============================================================================
# INTERFACE CONTROL
# =============================================================================

def parse_args() -> GANConfig:
    config = GANConfig()
    parser = argparse.ArgumentParser(
        description="Train a WGAN-GP on CelebA Faces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--g-lr", type=float, default=config.g_lr)
    parser.add_argument("--d-lr", type=float, default=config.d_lr)
    parser.add_argument("--latent-dim", type=int, default=config.latent_dim)
    parser.add_argument("--n-critic", type=int, default=config.n_critic)
    parser.add_argument("--gp-weight", type=float, default=config.gp_weight)
    parser.add_argument("--no-amp", action="store_true", default=not config.use_amp)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    args = parser.parse_args()

    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.g_lr = args.g_lr
    config.d_lr = args.d_lr
    config.latent_dim = args.latent_dim
    config.n_critic = args.n_critic
    config.gp_weight = args.gp_weight
    config.use_amp = not args.no_amp
    config.seed = args.seed
    config.num_workers = args.num_workers
    return config


if __name__ == "__main__":
    runtime_config = parse_args()
    train(runtime_config)