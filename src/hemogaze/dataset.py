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

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .features import roi_bbox, roi_mask

# ImageNet statistics, since every backbone here is pretrained on it.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool, strong_aug: bool = False):
    """Augmentation for a pallor task.

    Geometry is augmented freely; colour is barely touched. Pallor *is* the
    colour signal, so heavy brightness/contrast/hue jitter would train the model
    to ignore the only feature that carries the diagnosis. The mild jitter that
    remains stands in for real phone-to-phone exposure differences, which is the
    variation we actually want invariance to.

    ``strong_aug`` adds RandomResizedCrop, flips and wider rotation. Its purpose
    is not regularisation but shortcut removal: varying position and scale stops
    the absolute geometry of the hand-traced outline from being a reliable cue.
    Test-time transforms stay deterministic.
    """
    eval_tf = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if not train:
        return transforms.Compose(eval_tf)

    train_tf = []
    if strong_aug:
        # scale floor is 0.5, not 0.3: the ROI covers under a third of the
        # frame, so very small crops frequently contain no tissue at all. Those
        # views carry a label the image cannot support, and that added label
        # noise would be indistinguishable from the effect being measured.
        train_tf += [transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0),
                                                  ratio=(0.75, 1.75)),
                     transforms.RandomHorizontalFlip(),
                     transforms.RandomVerticalFlip(),
                     transforms.RandomRotation(20)]
    else:
        train_tf += [transforms.Resize((image_size, image_size)),
                     transforms.RandomHorizontalFlip(),
                     transforms.RandomRotation(10)]
    train_tf += [transforms.ColorJitter(brightness=0.1, contrast=0.1),
                 transforms.ToTensor(),
                 transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(train_tf)


NEUTRAL_FILL = 128          # mid-grey, used at evaluation time


def fill_background(img: Image.Image, train: bool,
                    rng: np.random.Generator | None = None,
                    black_threshold: int = 20) -> Image.Image:
    """Replace the segmentation background with something uninformative.

    Grad-CAM on the first CP-AnemiC experiment showed the network putting most
    of its attention on the black region around the conjunctiva rather than on
    the tissue. That region is an artefact of hand segmentation, traced by
    different people at different hospitals, so attending to it is a site
    fingerprint and not a clinical signal.

    Cropping to the ROI does not remove it here (the ROI is a crescent whose
    bounding box is the whole frame), so instead the background is made
    worthless to look at: a random solid colour plus noise, redrawn every time
    the image is seen during training. A feature that changes every epoch cannot
    carry a stable label association, so the network has to fall back on the
    tissue. At evaluation the fill is a fixed mid-grey, because a test-time
    prediction must not depend on a random draw.

    Honest limit: this kills the *interior* of the background as a cue but not
    the tissue outline itself. The boundary between tissue and fill still traces
    the same hand-drawn shape. This weakens the shortcut; it does not prove it
    gone.
    """
    arr = np.asarray(img).copy()
    bg = ~roi_mask(arr, black_threshold)
    if not bg.any():
        return img
    if train:
        rng = rng or np.random.default_rng()
        base = rng.integers(0, 256, size=3)
        noise = rng.normal(0, 12.0, size=(int(bg.sum()), 3))
        arr[bg] = np.clip(base + noise, 0, 255).astype(np.uint8)
    else:
        arr[bg] = NEUTRAL_FILL
    return Image.fromarray(arr)


def to_silhouette(img: Image.Image, black_threshold: int = 20) -> Image.Image:
    """Throw away every pixel value and keep only the shape of the ROI.

    This is a **positive control**, not an ablation. Two attempts to remove the
    segmentation shortcut and watch the model get worse both failed: cropping to
    the ROI does nothing on a crescent, and repainting the background destroyed
    training because the background is 72% of the frame. Worse, repainting
    attacked the wrong variable -- the background is uniform black in all 710
    images, so its colour cannot distinguish a hospital; only the traced outline
    varies.

    So instead of subtracting the cue, measure it on its own. A model trained on
    white-on-black silhouettes has access to the hand-drawn shape and to nothing
    else: no pallor, no colour, no texture. If it reaches the AUROC of the
    full-image model, the shortcut hypothesis is demonstrated directly. If it
    sits at chance, the hypothesis is dead and the Grad-CAM reading was
    over-interpretation.
    """
    mask = roi_mask(np.asarray(img), black_threshold)
    return Image.fromarray(np.repeat((mask * 255).astype(np.uint8)[:, :, None],
                                     3, axis=2))


class ConjunctivaDataset(Dataset):
    """Rows of an already-split metadata frame -> (image tensor, target).

    ``target_col`` is "label" for the screening task and "hemoglobin" for the
    regression variant; both return a float scalar so the same training loop
    serves either with only the loss swapped.
    """

    def __init__(self, df: pd.DataFrame, data_dir: str | Path, image_size: int,
                 train: bool, target_col: str = "label",
                 crop_to_roi: bool = False, strong_aug: bool = False,
                 randomise_background: bool = False,
                 silhouette_only: bool = False, seed: int = 42):
        if target_col not in df.columns:
            raise ValueError(f"Metadata has no {target_col!r} column.")
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.target_col = target_col
        self.crop_to_roi = crop_to_roi
        self.randomise_background = randomise_background
        self.silhouette_only = silhouette_only
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.tf = build_transforms(image_size, train, strong_aug)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        path = self.data_dir / str(row["image_id"])
        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.crop_to_roi:
            top, left, bottom, right = roi_bbox(np.asarray(img))
            img = img.crop((left, top, right, bottom))
        if self.silhouette_only:
            img = to_silhouette(img)
        elif self.randomise_background:
            img = fill_background(img, train=self.train, rng=self.rng)
        x = self.tf(img)
        y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return x, y


def make_dataset(df: pd.DataFrame, data_dir: str | Path, image_size: int,
                 train: bool, target_col: str = "label",
                 crop_to_roi: bool = False, strong_aug: bool = False,
                 randomise_background: bool = False,
                 silhouette_only: bool = False,
                 seed: int = 42) -> ConjunctivaDataset:
    """Kept as a function so call sites read the same as before."""
    return ConjunctivaDataset(df, data_dir, image_size, train, target_col,
                              crop_to_roi, strong_aug, randomise_background,
                              silhouette_only, seed)
