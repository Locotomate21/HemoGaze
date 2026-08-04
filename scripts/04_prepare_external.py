#!/usr/bin/env python
"""Step 4: adapt Eyes-Defy-Anemia into the schema, for EXTERNAL validation.

CP-AnemiC answers "does this work at another hospital in Ghana". This dataset
answers the harder question nobody in the literature reports: does a model
trained on Ghanaian children work on Italian and Indian adults, with different
phones, different clinicians, and a different segmentation protocol.

The distribution ships as:

    dataset anemia/
        India/   1/ .. 95/    each: <photo>.jpg + _palpebral.png
                               + _forniceal.png + _forniceal_palpebral.png
        Italy/   1/ .. 123/   same
        India.xlsx, Italy.xlsx    Number, Hgb, Gender, Age, Note

Three decisions, each forced by an audit rather than by preference.

**The palpebral variant, not forniceal.** CP-AnemiC images are palpebral
conjunctiva, so this is the like-for-like comparison. Using the forniceal or
combined crop would change the anatomy between train and test and confound the
whole point.

**61 of 218 palpebral PNGs are truncated, and they are excluded.** PIL recovers
60 of them with LOAD_TRUNCATED_IMAGES, and the recovered images look plausible:
the fraction of rows containing tissue is 0.28 against 0.31 for intact files. But
their colour is systematically shifted -- mean redness 0.20 vs 0.25, r_mean 171
vs 182, Mann-Whitney p < 0.001 -- while the patients' real hemoglobin is
indistinguishable (12.88 vs 12.61 g/dL, p = 0.478). So the corruption makes
tissue look paler without those people being anemic. Including them would not add
noise, it would add bias pointing at "anemic" in exactly the sample meant to
judge the model. 155 usable images remain.

**The WHO cutoff is not inherited.** These are adults aged 19-88, so anemia is
Hb < 12 g/dL for women and < 13 for men, not the 11 used for children 6-59
months. Applying the child cutoff here would report 25.2% prevalence instead of
41.9%. This is the reason the project moved to regressing hemoglobin: the model
predicts g/dL and the threshold is applied afterwards, per person.

Usage:
    python scripts/04_prepare_external.py --src "C:/path/to/dataset anemia"
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemogaze import splits as S

COUNTRIES = ("India", "Italy")
# WHO haemoglobin thresholds for non-pregnant adults, g/dL.
ADULT_CUTOFF = {"F": 12.0, "M": 13.0}
SOURCE = ("Eyes-Defy-Anemia (Dimauro et al.), conjunctiva images from Italian "
          "and Indian patients with paired haemoglobin")


def palpebral_path(folder: Path) -> Path | None:
    """The palpebral crop, excluding the forniceal_palpebral combination whose
    filename also ends in _palpebral.png."""
    hits = [p for p in folder.glob("*_palpebral.png")
            if "forniceal" not in p.name]
    return hits[0] if hits else None


def is_intact(path: Path) -> bool:
    """True if the PNG decodes without truncation. Checked with the permissive
    loader OFF, because that is the whole point of the test."""
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unzipped 'dataset anemia' folder")
    ap.add_argument("--out", default="data/eyes-defy")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    src, out = Path(args.src), Path(args.out)
    rows, dropped = [], {"truncated": 0, "no_palpebral": 0, "no_hb": 0}

    for country in COUNTRIES:
        sheet = src / country / f"{country}.xlsx"
        if not sheet.exists():
            sys.exit(f"Missing {sheet}. Point --src at the unzipped folder.")
        meta = pd.read_excel(sheet).set_index("Number")

        for folder in sorted((src / country).iterdir()):
            if not folder.is_dir():
                continue
            path = palpebral_path(folder)
            if path is None:
                dropped["no_palpebral"] += 1
                continue
            if not is_intact(path):
                dropped["truncated"] += 1
                continue
            try:
                row = meta.loc[int(folder.name)]
            except (KeyError, ValueError):
                dropped["no_hb"] += 1
                continue
            hb = pd.to_numeric(row.get("Hgb"), errors="coerce")
            sex = str(row.get("Gender", "")).strip().upper()[:1]
            if not np.isfinite(hb) or sex not in ADULT_CUTOFF:
                dropped["no_hb"] += 1
                continue

            image_id = f"images/{country}_{folder.name}.png"
            rows.append({
                "image_id": image_id,
                # One photograph per patient, as in CP-AnemiC.
                "patient_id": f"{country}_{folder.name}",
                # Country is the site axis here: two collection centres,
                # different phones and different clinicians.
                "site": country,
                "hemoglobin": float(hb),
                "sex": "Female" if sex == "F" else "Male",
                "age_years": pd.to_numeric(row.get("Age"), errors="coerce"),
                "cutoff": ADULT_CUTOFF[sex],
                "label": int(hb < ADULT_CUTOFF[sex]),
                "_src": path,
            })

    if not rows:
        sys.exit("No usable images found; check --src.")
    df = pd.DataFrame(rows)

    (out / "images").mkdir(parents=True, exist_ok=True)
    for _, r in df.iterrows():
        shutil.copy2(r["_src"], out / r["image_id"])
    meta_out = df.drop(columns=["_src"])
    S._check_columns(meta_out)
    meta_out.to_csv(out / "metadata.csv", index=False)

    (out / "PREPARED.json").write_text(json.dumps({
        "source": str(src), "dataset": SOURCE,
        "n_images": int(len(meta_out)),
        "n_dropped": dropped,
        "sites": meta_out["site"].value_counts().to_dict(),
        "prevalence_adult_cutoff": float(meta_out["label"].mean()),
        "label_rule": "hemoglobin < 12 g/dL (women) or < 13 (men), WHO adults",
        "why_truncated_excluded": (
            "61 palpebral PNGs are truncated. They decode with "
            "LOAD_TRUNCATED_IMAGES but their colour is systematically paler "
            "(redness 0.20 vs 0.25, p<0.001) while the patients' real "
            "haemoglobin is indistinguishable (12.88 vs 12.61, p=0.478). The "
            "corruption biases toward 'anemic' without those people being "
            "anemic -- fatal in a set whose job is to judge the model."),
        "intended_use": (
            "EXTERNAL VALIDATION ONLY. Adults 19-88 against CP-AnemiC's "
            "children 6-59 months, so the WHO cutoff differs and is stored "
            "per row. Evaluate a Ghana-trained hemoglobin regressor here; do "
            "not train on it."),
    }, indent=2))

    print(f"wrote {len(meta_out)} images -> {out}")
    print(f"dropped: {dropped}")
    print(f"sites: {meta_out['site'].value_counts().to_dict()}")
    print(f"hemoglobin: mean {meta_out['hemoglobin'].mean():.2f} g/dL "
          f"(CP-AnemiC: 10.36) | age {meta_out['age_years'].min():.0f}-"
          f"{meta_out['age_years'].max():.0f} years")
    print(f"prevalence at the adult cutoff: {meta_out['label'].mean():.3f} "
          f"(it would be {(meta_out['hemoglobin'] < 11).mean():.3f} at the "
          f"child cutoff -- which is why the threshold is not inherited)")
    print("\nEXTERNAL VALIDATION ONLY -- do not train on this.")


if __name__ == "__main__":
    main()
