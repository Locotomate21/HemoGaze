#!/usr/bin/env python
"""Generate a SYNTHETIC stand-in for CP-AnemiC so the pipeline can be run and
tested end to end without any patient data.

This is a plumbing tool, not part of the pipeline (hence no number prefix). It
exists because the honest core of this repo -- splits, leakage guards, baselines,
the cross-site gap -- must be exercised on data whose ground truth we control.

What it fabricates, on purpose:

* a weak *real* signal: anemic conjunctiva images are drawn slightly paler
  (lower red channel, lower saturation) than non-anemic ones;
* a strong *site* confounder: each site gets its own colour cast and its own
  anemia prevalence, mimicking ten facilities with ten different phones and
  catchment populations;
* two images (two eyes) per patient, which is what makes patient-level
  splitting non-optional.

So a correct pipeline should score decently on a patient-level split, clearly
worse leave-one-site-out, and the audit in ``00_eda_baseline.py`` should light up
the imbalanced sites. Any *result* from this data is a test of the code, never a
finding about anemia -- every artifact written here is stamped SYNTHETIC.

Usage:
    python scripts/make_synthetic_data.py --out data/synthetic
    python scripts/00_eda_baseline.py --config config/synthetic.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Ten sites, as in the real dataset. Each entry is
# (site name, anemia prevalence, colour cast as RGB multipliers).
# Two sites are deliberately near-single-class so the confounder audit has
# something true to flag.
SITES = [
    ("SITE_A", 0.55, (1.00, 1.00, 1.00)),
    ("SITE_B", 0.48, (1.06, 0.97, 0.95)),
    ("SITE_C", 0.62, (0.94, 1.02, 1.05)),
    ("SITE_D", 0.35, (1.02, 1.04, 0.96)),
    ("SITE_E", 0.50, (0.97, 0.98, 1.06)),
    ("SITE_F", 0.44, (1.08, 1.01, 0.99)),
    ("SITE_G", 0.58, (0.96, 0.95, 1.02)),
    ("SITE_H", 0.40, (1.03, 0.99, 1.04)),
    ("SITE_I", 0.91, (1.05, 1.06, 1.01)),   # almost all anemic  -> should flag
    ("SITE_J", 0.09, (0.93, 0.96, 0.98)),   # almost none anemic -> should flag
]


def synth_image(rng: np.random.Generator, anemic: bool, cast, size: int = 96
                ) -> np.ndarray:
    """A crude conjunctiva stand-in: a reddish blob on a dark background.

    Pallor is encoded exactly as it is clinically read -- less red, less
    saturated -- but with enough per-image noise that the task stays hard.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size / 2 + rng.normal(0, 3), size / 2 + rng.normal(0, 3)
    rx, ry = size * 0.38, size * 0.20
    roi = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0

    # Base conjunctiva colour, paler when anemic. The offset is deliberately
    # smaller than the per-image variation: a stand-in dataset on which the
    # colour baseline scores 0.99 would hide exactly the bugs we want to catch.
    base = np.array([175.0, 95.0, 95.0])
    if anemic:
        base = base - np.array([13.0, 2.0, 3.0])
    base = base + rng.normal(0, 19.0, size=3)          # per-image variation

    img = np.full((size, size, 3), 28.0)                # dark surround
    img[roi] = base
    img *= np.asarray(cast)                             # per-site camera cast
    img += rng.normal(0, 6.0, size=img.shape)           # sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--patients-per-site", type=int, default=36)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    rows = []
    for site, prevalence, cast in SITES:
        for p in range(args.patients_per_site):
            patient_id = f"{site}_P{p:03d}"
            anemic = bool(rng.random() < prevalence)
            # Hemoglobin consistent with the label and the WHO cutoff of 11 g/dL.
            hb = float(rng.normal(9.4, 0.9) if anemic else rng.normal(12.3, 1.0))
            hb = float(np.clip(hb, 5.0, 16.0))
            for eye in ("L", "R"):
                image_id = f"images/{patient_id}_{eye}.png"
                Image.fromarray(synth_image(rng, anemic, cast)).save(out / image_id)
                rows.append(dict(
                    image_id=image_id, patient_id=patient_id, site=site,
                    label=int(anemic), hemoglobin=round(hb, 1),
                    age_months=int(rng.integers(6, 60)),
                    sex=str(rng.choice(["M", "F"])),
                ))

    df = pd.DataFrame(rows)
    df.to_csv(out / "metadata.csv", index=False)
    (out / "SYNTHETIC.json").write_text(json.dumps({
        "synthetic": True,
        "seed": args.seed,
        "generator": "scripts/make_synthetic_data.py",
        "warning": ("Fabricated data for pipeline testing only. Any metric "
                    "computed on it describes the code, not anemia, and must "
                    "never be quoted as a result."),
    }, indent=2))

    print(f"wrote {len(df)} synthetic images for {df['patient_id'].nunique()} "
          f"patients across {df['site'].nunique()} sites -> {out}")
    print(f"overall anemic prevalence: {df['label'].mean():.3f}")
    print("[SYNTHETIC] results from this data test the pipeline, nothing more.")


if __name__ == "__main__":
    main()
