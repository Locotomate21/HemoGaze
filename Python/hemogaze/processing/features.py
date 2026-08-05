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


def roi_mask(img: np.ndarray, black_threshold: int = 20) -> np.ndarray:
    """Boolean ROI mask that drops a segmented image's black background.

    CP-AnemiC ships each conjunctiva already cut out and pasted on pure black,
    and the background is 55-87% of the frame depending on how the strip was
    traced. Averaging over the whole image therefore measures *how much black
    the crop happens to contain* far more than it measures pallor: on real
    images `r_mean` moves from ~197 inside the ROI to ~70 over the full frame.
    Worse, that background fraction varies by 32 percentage points between
    images, so it would act as a strong spurious feature correlated with
    whoever traced the outline.

    On an unsegmented photo almost nothing is pure black, so this returns
    approximately all-True and is a no-op -- safe to apply unconditionally.
    """
    return img.max(axis=-1) > black_threshold


def roi_bbox(img: np.ndarray, black_threshold: int = 20) -> tuple[int, int, int, int]:
    """Tight bounding box of the ROI as (top, left, bottom, right), end-exclusive.

    Used to strip the framing before a CNN sees the image. In the first
    CP-AnemiC experiment the network was fed the full frame, 73% of which is
    black, and Grad-CAM showed it attending to that background and to the
    silhouette of the hand-traced crop rather than to the tissue. The outline
    was traced by different people at different hospitals, so it is a site
    fingerprint -- exactly the confounder the colour baseline never had access
    to, because averaging over a mask discards geometry.

    Cropping does not change the colour baseline at all (its statistics already
    ignore every background pixel), which is what makes it a matched
    intervention rather than merely a different one.

    **Measured to be useless on CP-AnemiC, and kept for the record.** A
    conjunctiva is a thin curved crescent spanning the frame diagonally, so its
    bounding box is essentially the whole image: over 80 real images the
    background fraction was a median 72% before cropping and 72% after. It works
    as intended on a compact ROI, which is why the tests pass -- it just does not
    describe this dataset. Use ``randomise_background`` instead here.

    Falls back to the full frame if the mask is empty, so a fully-black or
    unsegmented image cannot produce a degenerate crop.
    """
    mask = roi_mask(img, black_threshold)
    if not mask.any():
        return 0, 0, img.shape[0], img.shape[1]
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


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
