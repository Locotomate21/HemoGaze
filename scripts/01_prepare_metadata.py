#!/usr/bin/env python
"""Step 1: turn the CP-AnemiC download into the schema the pipeline expects.

The Kaggle distribution ships as:

    CP-AnemiC dataset/
        Anemia_Data_Collection_Sheet.xlsx   710 rows of clinical metadata
        Anemic/          Image_001.png ...  424 files
        Non-anemic/      Image_003.png ...  286 files

and this script maps it onto `data/cp-anemic/` with the four required columns
plus the optional ones, verifying along the way that the spreadsheet and the
folders actually agree.

Column mapping:

    IMAGE_ID      -> image_id      (plus the .png extension and folder prefix)
    HOSPITAL      -> site          the confounder axis: ten facilities
    HB_LEVEL      -> hemoglobin    g/dL, the reference standard
    HB_LEVEL < 11 -> label         WHO cutoff for children under five
    Age(Months)   -> age_months
    GENDER        -> sex
    Severity      -> severity      kept for the future multi-grade variant
    REGION        -> region        kept for a coarser cross-site analysis

## The patient_id problem, stated plainly

**CP-AnemiC ships no patient identifier.** The spreadsheet has one row per
image, the filenames are a flat sequence `Image_001 ... Image_710` with no eye
or subject suffix, and nothing else groups rows by child. So this script sets
`patient_id = image_id`, i.e. it assumes **one image per child**.

That assumption is load-bearing. If the source study actually photographed both
eyes of ~355 children, then a "patient-level" split built on this metadata is
really an image-level split, the two eyes of one child land on opposite sides,
and every in-distribution number is inflated by leakage. Nothing in the
delivered files lets us detect that, so the assumption is recorded in
`data/cp-anemic/PREPARED.json` and printed on every run rather than buried.

Verify it against the source publication before quoting any patient-level
result. If it turns out to be wrong, only the leave-one-site-out numbers
survive.

Usage:
    python scripts/01_prepare_metadata.py --src "C:/path/to/CP-AnemiC dataset"
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemogaze import splits as S

SHEET = "Anemia_Data_Collection_Sheet.xlsx"
FOLDERS = {"Anemic": 1, "Non-anemic": 0}
WHO_CUTOFF_G_DL = 11.0        # children 6-59 months


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unzipped CP-AnemiC folder")
    ap.add_argument("--out", default="data/cp-anemic")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    sheet = src / SHEET
    if not sheet.exists():
        sys.exit(f"No {SHEET} under {src}. Point --src at the unzipped folder.")

    df = pd.read_excel(sheet)
    print(f"spreadsheet: {len(df)} rows, {df['IMAGE_ID'].nunique()} unique IMAGE_ID")

    # --- reconcile the spreadsheet against the two image folders -------------
    on_disk = {}
    for folder, folder_label in FOLDERS.items():
        d = src / folder
        if not d.exists():
            sys.exit(f"Missing image folder {d}.")
        for p in d.glob("*.png"):
            if p.stem in on_disk:
                sys.exit(f"{p.stem} appears in both folders -- contradictory "
                         f"labels, stop and inspect the download.")
            on_disk[p.stem] = (folder, folder_label)
    print(f"images on disk: {len(on_disk)}")

    missing = set(df["IMAGE_ID"]) - set(on_disk)
    orphans = set(on_disk) - set(df["IMAGE_ID"])
    if missing:
        sys.exit(f"{len(missing)} rows have no image file, e.g. {sorted(missing)[:5]}")
    if orphans:
        sys.exit(f"{len(orphans)} images have no metadata row, e.g. {sorted(orphans)[:5]}")

    df["folder"] = df["IMAGE_ID"].map(lambda i: on_disk[i][0])
    df["folder_label"] = df["IMAGE_ID"].map(lambda i: on_disk[i][1])
    df["label"] = (df["HB_LEVEL"] < WHO_CUTOFF_G_DL).astype(int)

    # The folder name is an independent copy of the label. If it disagrees with
    # the WHO cutoff applied to the measured hemoglobin, we do not get to pick
    # whichever is convenient -- we stop.
    clash = df[df["label"] != df["folder_label"]]
    if len(clash):
        sys.exit(f"{len(clash)} images where the folder disagrees with "
                 f"Hb < {WHO_CUTOFF_G_DL}: {clash['IMAGE_ID'].tolist()[:10]}. "
                 f"Resolve against the source before training.")
    print(f"label check: folder and Hb < {WHO_CUTOFF_G_DL} g/dL agree on all "
          f"{len(df)} images")

    # --- write the schema ----------------------------------------------------
    meta = pd.DataFrame({
        "image_id": "images/" + df["IMAGE_ID"] + ".png",
        # See the module docstring: no patient identifier exists in this dataset.
        "patient_id": df["IMAGE_ID"],
        "site": df["HOSPITAL"],
        "label": df["label"],
        "hemoglobin": df["HB_LEVEL"],
        "age_months": df["Age(Months)"],
        "sex": df["GENDER"],
        "severity": df["Severity"],
        "region": df["REGION"],
    })
    S._check_columns(meta)

    (out / "images").mkdir(parents=True, exist_ok=True)
    for image_id, folder in zip(df["IMAGE_ID"], df["folder"]):
        shutil.copy2(src / folder / f"{image_id}.png",
                     out / "images" / f"{image_id}.png")
    meta.to_csv(out / "metadata.csv", index=False)

    (out / "PREPARED.json").write_text(json.dumps({
        "source": str(src),
        "n_images": int(len(meta)),
        "n_sites": int(meta["site"].nunique()),
        "prevalence": float(meta["label"].mean()),
        "label_rule": f"hemoglobin < {WHO_CUTOFF_G_DL} g/dL (WHO, 6-59 months)",
        "patient_id_assumption": (
            "ONE IMAGE PER CHILD. CP-AnemiC ships no patient identifier, so "
            "patient_id = image_id. If the source study photographed both eyes "
            "of each child, the patient-level split is leaky and only the "
            "leave-one-site-out results stand. Verify against the publication."
        ),
    }, indent=2))

    print(f"\nwrote {out/'metadata.csv'} and {len(meta)} images -> {out/'images'}")
    print(f"sites: {meta['site'].nunique()} | prevalence: {meta['label'].mean():.3f}")
    print("\n[!] patient_id = image_id (one image per child ASSUMED; this "
          "dataset ships no patient identifier). See data/cp-anemic/"
          "PREPARED.json and verify against the source publication.")
    print("\nNext: python scripts/00_eda_baseline.py --config config/default.yaml")


if __name__ == "__main__":
    main()
