
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms


def get_cifar10_loaders(project_dir=".", download=True):
    project_dir = Path(project_dir)
    config_dir = project_dir / "config"

    # Load shared configuration
    with open(config_dir / "preprocessing_config.json") as f:
        config = json.load(f)

    with np.load(config_dir / "split_indices.npz") as data:
        train_indices = data["train_indices"]
        val_indices = data["val_indices"]

    # Create normalization transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["mean"],
            std=config["std"]
        )
    ])

    # Load CIFAR-10
    train_full = CIFAR10(
        root=str(project_dir / "data"),
        train=True,
        download=download,
        transform=transform
    )

    test_dataset = CIFAR10(
        root=str(project_dir / "data"),
        train=False,
        download=download,
        transform=transform
    )

    # Reuse identical dataset partitions
    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(train_full, val_indices)

    batch_size = config["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader
