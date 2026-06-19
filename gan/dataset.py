import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms
from torchvision.datasets import CelebA

from config import GANConfig

# =============================================================================
# Unsupervised Wrapper
# =============================================================================

class UnsupervisedCelebA(Dataset):
    """
    Wraps Torchvision's CelebA dataset to safely drop unused target structures,
    returning ONLY the image tensor to prevent DataLoader collation errors.
    """
    def __init__(self, celeba_dataset: CelebA) -> None:
        super().__init__()
        self.dataset = celeba_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Torchvision returns (image, targets) or (image, None)
        # We cleanly extract just the processed image tensor
        img, _ = self.dataset[idx]
        return img


# =============================================================================
# Dataloader Creation Pipeline
# =============================================================================

def get_celeba_transforms(config: GANConfig, is_train: bool) -> transforms.Compose:
    transform_list = [
        transforms.CenterCrop(178),
        transforms.Resize((config.image_size, config.image_size)),
    ]

    if config.in_channels == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    transform_list.append(transforms.ToImage())
    transform_list.append(transforms.ToDtype(torch.float32, scale=True))

    if is_train:
        transform_list.insert(0, transforms.RandomHorizontalFlip(p=0.5))

    return transforms.Compose(transform_list)


def create_dataloaders(config: GANConfig) -> tuple[DataLoader, DataLoader]:
    print(f"[DATA] Initializing CelebA datasets out of: {config.data_cache_dir.resolve()}")

    # 1. Instantiate the raw Torchvision dataset blocks
    # We pass target_type="attr" explicitly so Torchvision initializes standard structures internally
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

    # 2. Intercept them with our custom unsupervised wrapper
    train_set = UnsupervisedCelebA(raw_train)
    val_set = UnsupervisedCelebA(raw_val)

    # 3. Stream through standard DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        persistent_workers=config.num_workers > 0,
    )

    print(f"[DATA] Train dataset ready: {len(train_set)} samples ({len(train_loader)} batches)")
    print(f"[DATA] Val dataset ready:   {len(val_set)} samples ({len(val_loader)} batches)")

    return train_loader, val_loader