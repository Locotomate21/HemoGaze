"""Torch Dataset for CP-AnemiC conjunctiva images.

This module is the torch boundary of the package: it imports torch at module
level, and nothing in ``splits`` / ``metrics`` / ``features`` / ``baselines``
imports it, so the honest core stays runnable with no deep-learning stack.
``__init__`` deliberately does not import it either.

The Dataset class is defined at module level rather than built inside a factory,
because a class defined inside a function cannot be pickled -- and on Windows
(spawn-based multiprocessing) that breaks any DataLoader with num_workers > 0.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet statistics, since every backbone here is pretrained on it.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool):
    """Augmentation for a pallor task.

    Geometry is augmented freely; colour is barely touched. Pallor *is* the
    colour signal, so heavy brightness/contrast/hue jitter would train the model
    to ignore the only feature that carries the diagnosis. The mild jitter that
    remains stands in for real phone-to-phone exposure differences, which is the
    variation we actually want invariance to.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ConjunctivaDataset(Dataset):
    """Rows of an already-split metadata frame -> (image tensor, target).

    ``target_col`` is "label" for the screening task and "hemoglobin" for the
    regression variant; both return a float scalar so the same training loop
    serves either with only the loss swapped.
    """

    def __init__(self, df: pd.DataFrame, data_dir: str | Path, image_size: int,
                 train: bool, target_col: str = "label"):
        if target_col not in df.columns:
            raise ValueError(f"Metadata has no {target_col!r} column.")
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.target_col = target_col
        self.tf = build_transforms(image_size, train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        path = self.data_dir / str(row["image_id"])
        with Image.open(path) as im:
            img = im.convert("RGB")
        x = self.tf(img)
        y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return x, y


def make_dataset(df: pd.DataFrame, data_dir: str | Path, image_size: int,
                 train: bool, target_col: str = "label") -> ConjunctivaDataset:
    """Kept as a function so call sites read the same as before."""
    return ConjunctivaDataset(df, data_dir, image_size, train, target_col)
