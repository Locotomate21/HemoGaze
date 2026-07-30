"""Cheap, interpretable colour features from the conjunctiva ROI.

The clinical sign is *pallor* -- loss of redness. So a three-line colour model
is a genuinely strong baseline, and if a heavy CNN cannot clearly beat it, that
is a finding worth reporting, not something to hide.

These functions take numpy image arrays (H, W, 3) in RGB, 0-255, so they are
testable without any image files present.
"""
from __future__ import annotations

import numpy as np


def rgb_to_hsv_np(img: np.ndarray) -> np.ndarray:
    """Vectorised RGB(0-255) -> HSV(0-1). Avoids a hard OpenCV dependency for
    the baseline so it runs anywhere."""
    arr = img.astype(np.float64) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(-1)
    mn = arr.min(-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff != 0
    # hue
    rc = np.where(mask, (mx - r) / np.where(diff == 0, 1, diff), 0)
    gc = np.where(mask, (mx - g) / np.where(diff == 0, 1, diff), 0)
    bc = np.where(mask, (mx - b) / np.where(diff == 0, 1, diff), 0)
    h = np.where(mx == r, bc - gc, np.where(mx == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    s = np.where(mx == 0, 0, diff / np.where(mx == 0, 1, mx))
    v = mx
    return np.stack([h, s, v], axis=-1)


def color_features(img: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Extract pallor-relevant colour statistics over the ROI.

    img:  (H, W, 3) RGB uint8-ish
    mask: (H, W) bool of the conjunctiva ROI. If None, whole image is used
          (you should pass a real ROI mask in practice).
    """
    if mask is None:
        mask = np.ones(img.shape[:2], dtype=bool)
    px = img[mask].astype(np.float64)
    if px.size == 0:
        raise ValueError("Empty ROI mask.")
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    hsv = rgb_to_hsv_np(px.reshape(-1, 1, 3)).reshape(-1, 3)
    eps = 1e-6
    feats = {
        "r_mean": r.mean(), "g_mean": g.mean(), "b_mean": b.mean(),
        "r_std": r.std(), "g_std": g.std(),
        "rg_ratio": r.mean() / (g.mean() + eps),
        "redness": (r.mean() - g.mean()) / (r.mean() + g.mean() + eps),
        "r_p10": np.percentile(r, 10), "r_p90": np.percentile(r, 90),
        "sat_mean": hsv[:, 1].mean(),
        "val_mean": hsv[:, 2].mean(),
    }
    return {k: float(v) for k, v in feats.items()}


FEATURE_ORDER = ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "rg_ratio",
                 "redness", "r_p10", "r_p90", "sat_mean", "val_mean"]


def features_to_vector(feats: dict) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float64)
