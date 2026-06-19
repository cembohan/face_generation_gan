"""
Face Generation Script
──────────────────────
Loads a trained VAE checkpoint and generates a grid of random faces.
Outputs a single PNG file with faces arranged in a 5×5 grid.
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from config import VAEConfig
from model import VAE


def load_model(checkpoint_path: str, device: torch.device) -> tuple[VAE, dict]:
    """Load VAE model from checkpoint."""
    print(f"[GEN] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Reconstruct config from saved checkpoint info
    ckpt_config = checkpoint["config"]
    config = VAEConfig(
        latent_dim=ckpt_config["latent_dim"],
        encoder_channels=ckpt_config["encoder_channels"],
        decoder_channels=ckpt_config["decoder_channels"],
        image_size=ckpt_config["image_size"],
        in_channels=ckpt_config["in_channels"],
    )

    model = VAE(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[GEN] Model loaded (latent_dim={config.latent_dim}, params={model.get_num_params():,})")
    if "epoch" in checkpoint:
        print(f"[GEN] Checkpoint from epoch {checkpoint['epoch']}, val_loss={checkpoint.get('val_loss', 'N/A')}")

    return model, ckpt_config


def generate_grid(
    model: VAE,
    device: torch.device,
    rows: int = 5,
    cols: int = 5,
    temperature: float = 1.0,
    image_size: int = 64,
    padding: int = 4,
    bg_value: int = 30,
) -> np.ndarray:
    """
    Generate a grid of faces from the VAE's prior.

    Args:
        model: Trained VAE model.
        device: torch device.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        temperature: Sampling temperature (1.0 = standard prior).
        image_size: Size of each generated face.
        padding: Pixels of padding between images.
        bg_value: Background pixel value (0-255) for the padding area.

    Returns:
        numpy array (H, W) uint8 representing the composite grid image.
    """
    n = rows * cols
    print(f"[GEN] Generating {n} faces (temperature={temperature:.2f})...")

    with torch.no_grad():
        samples = model.sample(n, device, temperature=temperature)

    # Convert to numpy, (N, 1, H, W) → (N, H, W), clamp and scale to uint8
    images = samples.cpu().squeeze(1).clamp(0, 1).numpy()
    images = (images * 255).astype(np.uint8)

    # Build the grid
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
        description="Generate random faces from a trained VAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/generated_faces.png",
        help="Output file path for the generated grid",
    )
    parser.add_argument("--rows", type=int, default=5, help="Number of rows in the grid")
    parser.add_argument("--cols", type=int, default=5, help="Number of columns in the grid")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.85,
        help="Sampling temperature (lower = sharper, higher = more diverse)",
    )
    parser.add_argument(
        "--num-grids",
        type=int,
        default=1,
        help="Number of grid images to generate in this run",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (None = random)")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GEN] Device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # ── Load Model ────────────────────────────────────────────────────────
    model, ckpt_config = load_model(args.checkpoint, device)
    image_size = ckpt_config.get("image_size", 64)

    # ── Generate ──────────────────────────────────────────────────────────
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    base_output = Path(args.output)

    for i in range(args.num_grids):
        grid = generate_grid(
            model, device,
            rows=args.rows,
            cols=args.cols,
            temperature=args.temperature,
            image_size=image_size,
        )

        # Handle unique paths to prevent overriding previous script runs
        if args.num_grids == 1:
            out_path = get_unique_path(base_output)
        else:
            # If generating multiple grids in a single command, keep them ordered sequentially
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