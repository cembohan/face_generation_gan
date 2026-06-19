"""
GAN Face Generation Script
───────────────────────────
Loads a trained Generator checkpoint and produces a grid of random faces.
Same interface as the VAE generation script for easy comparison.
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from config import GANConfig   # or you can define a minimal config class locally
from model import Generator


def load_generator(checkpoint_path: str, device: torch.device) -> tuple[Generator, GANConfig]:
    """Load Generator from a GAN checkpoint."""
    print(f"[GEN] Loading checkpoint: {checkpoint_path}")
    
    # Allowlist Path objects to satisfy PyTorch's strict weights_only check
    import pathlib
    torch.serialization.add_safe_globals([pathlib.PureWindowsPath, pathlib.WindowsPath])
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Reconstruct config from saved info (same structure as train_gan.py saved)
    ckpt_config = checkpoint["config"]
    # Create a minimal config object with the fields Generator needs
    config = GANConfig(
        latent_dim=ckpt_config["latent_dim"],
        gen_channels=ckpt_config["gen_channels"],
        image_size=ckpt_config["image_size"],
        in_channels=ckpt_config["in_channels"],
    )

    generator = Generator(config).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    print(f"[GEN] Generator loaded (latent_dim={config.latent_dim}, "
          f"params={sum(p.numel() for p in generator.parameters()):,})")
    if "epoch" in checkpoint:
        print(f"[GEN] Checkpoint from epoch {checkpoint['epoch']}, "
              f"val_w_dist={checkpoint.get('w_dist', 'N/A')}")

    return generator, config


def generate_grid(
    generator: Generator,
    device: torch.device,
    rows: int = 5,
    cols: int = 5,
    temperature: float = 1.0,
    image_size: int = 64,
    padding: int = 4,
    bg_value: int = 30,
) -> np.ndarray:
    """
    Generate a grid of faces from the GAN's prior (N(0, I) scaled by temperature).

    The generator outputs in [-1, 1] because of Tanh activation,
    so we rescale to [0, 255] for PNG.
    """
    n = rows * cols
    print(f"[GEN] Generating {n} faces (temperature={temperature:.2f})...")

    with torch.no_grad():
        z = torch.randn(n, generator.fc[0].in_features, device=device) * temperature
        samples = generator(z)

    # Convert: (N, 1, H, W) -> (N, H, W), then [-1,1] -> [0,1] -> [0,255]
    images = samples.cpu().squeeze(1)
    images = ((images + 1) / 2).clamp(0, 1).numpy()
    images = (images * 255).astype(np.uint8)

    # Build grid
    grid_h = rows * image_size + (rows + 1) * padding
    grid_w = cols * image_size + (cols + 1) * padding
    grid = np.full((grid_h, grid_w), bg_value, dtype=np.uint8)

    for idx in range(n):
        r = idx // cols
        c = idx % cols
        y = padding + r * (image_size + padding)
        x = padding + c * (image_size + padding)
        grid[y : y + image_size, x : x + image_size] = images[idx]

    return grid


def get_unique_path(base_path: Path) -> Path:
    """Finds a path that does not exist by appending a numerical suffix if necessary."""
    if not base_path.exists():
        return base_path
    
    output_dir = base_path.parent
    stem = base_path.stem
    suffix = base_path.suffix
    
    counter = 1
    while True:
        candidate = output_dir / f"{stem}_{counter:03d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random faces from a trained GAN generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_gan/gan_epoch0042_wdist0.5177.pt",
        help="Path to the GAN checkpoint (contains generator state)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs_gan/generated_faces_gan.png",
        help="Output file path",
    )
    parser.add_argument("--rows", type=int, default=5, help="Grid rows")
    parser.add_argument("--cols", type=int, default=5, help="Grid columns")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.85,
        help="Scaling factor for the latent noise (lower = sharper, higher = more diverse)",
    )
    parser.add_argument(
        "--num-grids",
        type=int,
        default=1,
        help="Number of grid images to generate in this run",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GEN] Device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Load generator
    generator, config = load_generator(args.checkpoint, device)
    image_size = config.image_size

    # Output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    base_output = Path(args.output)

    for i in range(args.num_grids):
        grid = generate_grid(
            generator, device,
            rows=args.rows,
            cols=args.cols,
            temperature=args.temperature,
            image_size=image_size,
        )

        # Unique path engine to respect existing files
        if args.num_grids == 1:
            out_path = get_unique_path(base_output)
        else:
            stem = base_output.stem
            suffix = base_output.suffix
            candidate_path = output_dir / f"{stem}_{i + 1:03d}{suffix}"
            out_path = get_unique_path(candidate_path)

        img = Image.fromarray(grid, mode="L")
        img.save(out_path, optimize=True)
        print(f"[GEN] Saved: {out_path.resolve()} ({grid.shape[1]}×{grid.shape[0]} px)")

    print(f"\n[GEN] Done! Generated {args.num_grids} grid(s) of {args.rows}×{args.cols} faces.")


if __name__ == "__main__":
    main()