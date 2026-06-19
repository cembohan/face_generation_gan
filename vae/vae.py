"""
VAE Face Generation Pipeline
═════════════════════════════

A production-grade Variational Autoencoder for generating faces,
trained on the LFW (Labeled Faces in the Wild) dataset.

Project Structure:
    config.py       Central configuration (all hyperparameters)
    dataset.py      LFW dataset loading, preprocessing, DataLoaders
    model.py        VAE architecture (Encoder + Decoder + reparameterization)
    train.py        Full training pipeline with TensorBoard logging
    generate.py     Face generation with grid output

Quick Start:
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Train the VAE
    python train.py --epochs 300 --batch-size 64 --latent-dim 128

    # 3. Monitor training
    tensorboard --logdir=runs

    # 4. Generate faces
    python generate.py --checkpoint checkpoints/best_model.pt --output outputs/generated_faces.png

    # 5. Generate multiple grids
    python generate.py --checkpoint checkpoints/best_model.pt --num-grids 5 --temperature 0.85

CLI Options (train.py):
    --epochs         Number of training epochs (default: 300)
    --batch-size     Batch size (default: 64)
    --lr             Learning rate (default: 1e-3)
    --latent-dim     Latent space dimensionality (default: 128)
    --kl-warmup      KL annealing warmup epochs (default: 30)
    --kl-strategy    KL annealing strategy: linear | cyclical
    --recon-loss     Reconstruction loss: mse | bce
    --scheduler      LR scheduler: cosine | plateau | none
    --no-amp         Disable mixed precision training
    --seed           Random seed (default: 42)
    --num-workers    DataLoader workers (default: 4)

CLI Options (generate.py):
    --checkpoint     Path to model checkpoint
    --output         Output PNG file path
    --rows           Grid rows (default: 5)
    --cols           Grid columns (default: 5)
    --temperature    Sampling temperature (default: 0.85)
    --num-grids      Number of grid images to generate
    --seed           Random seed for reproducible generation
"""
