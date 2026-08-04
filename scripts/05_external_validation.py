#!/usr/bin/env python
"""Step 5: evaluate a CP-AnemiC-trained model on Eyes-Defy-Anemia.

This is the question the literature on conjunctival pallor does not answer.
Every paper we checked, including the one that published CP-AnemiC, evaluates
with random or 5-fold splits inside a single dataset. That measures "does it
work on the hospitals it was trained on". This script measures whether a model
trained on Ghanaian children aged 6-59 months says anything true about Italian
and Indian adults aged 19-88, photographed with other phones and segmented by
other people.

Nothing is retrained and nothing is fitted here. Weights come from
``reports/runs_hb/`` and are applied as-is.

Three numbers are reported, and the order matters:

1. **Raw MAE.** What you would get by deploying the model unchanged. This is the
   headline and it is not allowed to be adjusted away.
2. **Bias.** CP-AnemiC has a mean Hb of 10.36 g/dL and this population sits at
   12.61, so a systematic offset near 2 g/dL is expected before any modelling
   question is asked. Reporting it separately keeps "wrong on average" from
   being confused with "uninformative".
3. **Bias-corrected MAE.** After subtracting the mean error, does the model
   still track variation between patients? This is *not* a deployable number --
   the correction uses the test labels, so it is a diagnostic, not a result. It
   separates "does not transfer at all" from "transfers with a fixed offset a
   calibration step could remove".

A fourth check comes free from regressing hemoglobin rather than a label: the
anemia classification is derived afterwards using each patient's own WHO cutoff
(12 g/dL for women, 13 for men), which a model trained on the child threshold of
11 could not have produced.

Usage:
    python scripts/05_external_validation.py \\
        --run reports/runs_hb/hb128_patient_level \\
        --external data/eyes-defy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemogaze import baselines as B
from hemogaze import metrics as M
from hemogaze.config import load_config
from hemogaze.features import FEATURE_ORDER, color_features, roi_mask


def colour_features_for(df: pd.DataFrame, data_dir: Path, threshold: int,
                        cache: Path) -> pd.DataFrame:
    if cache.exists():
        c = pd.read_csv(cache)
        if set(c["image_id"]) == set(df["image_id"]):
            out = c.set_index("image_id").loc[df["image_id"]]
            out.index = df.index
            return out
    from PIL import Image
    rows = []
    for image_id in df["image_id"]:
        with Image.open(data_dir / image_id) as im:
            img = np.asarray(im.convert("RGB"))
        rows.append({"image_id": image_id,
                     **color_features(img, roi_mask(img, threshold))})
    feats = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(cache, index=False)
    out = feats.set_index("image_id").loc[df["image_id"]]
    out.index = df.index
    return out


def predict_cnn(run: Path, df: pd.DataFrame, data_dir: Path) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    from hemogaze.dataset import make_dataset
    from hemogaze.model import build_model, load_weights

    cfg = load_config(run / "config.yaml")
    model = build_model(cfg.backbone, pretrained=False, dropout=cfg.dropout)
    load_weights(model, run / "model.pt", "cpu").eval()
    ds = make_dataset(df, data_dir, cfg.image_size, train=False,
                      target_col="hemoglobin",
                      crop_to_roi=cfg.crop_to_roi,
                      randomise_background=cfg.randomise_background,
                      silhouette_only=cfg.silhouette_only, seed=cfg.seed)
    preds = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=cfg.batch_size):
            preds.append(model(x).squeeze(1).numpy())
    return np.concatenate(preds)


def report(name: str, hb_true, hb_pred, hb_train, cutoff, spec_target=0.90):
    """Raw, then bias, then bias-corrected -- in that order, on purpose."""
    raw = M.regression_report(hb_true, hb_pred, y_train=hb_train)
    corrected = M.regression_report(hb_true, hb_pred - raw.bias, y_train=hb_train)
    y = (np.asarray(hb_true) < np.asarray(cutoff)).astype(int)
    cls = M.classification_report(y, M.classify_from_hb(hb_pred, cutoff),
                                  spec_target)
    print(f"\n  --- {name} ---")
    print(f"    raw            {raw.summary_line()}")
    print(f"    bias-corrected MAE={corrected.mae:.2f} g/dL  "
          f"({corrected.mae_vs_trivial:.2f}x trivial)   [diagnostic only: uses "
          f"the test labels]")
    print(f"    derived class  AUROC={cls.auroc:.3f}  AUPRC={cls.auprc:.3f}  "
          f"sens@spec{spec_target:.2f}={cls.sens_at_spec:.3f}")
    if raw.note:
        print(f"    [!] {raw.note}")
    return {"raw": raw.as_dict(), "bias_corrected_mae": corrected.mae,
            "bias_corrected_vs_trivial": corrected.mae_vs_trivial,
            "derived_classification": cls.as_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a trained regression run dir")
    ap.add_argument("--external", default="data/eyes-defy")
    ap.add_argument("--train-metadata", default="data/cp-anemic/metadata.csv")
    ap.add_argument("--out", default="reports/external")
    args = ap.parse_args()

    run, ext = Path(args.run), Path(args.external)
    df = pd.read_csv(ext / "metadata.csv")
    train_df = pd.read_csv(args.train_metadata)
    hb_true = df["hemoglobin"].to_numpy(dtype=float)
    hb_train = train_df["hemoglobin"].to_numpy(dtype=float)
    cutoff = df["cutoff"].to_numpy(dtype=float)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("===== EXTERNAL VALIDATION =====")
    print(f"train: {args.train_metadata}  n={len(train_df)}  "
          f"mean Hb {hb_train.mean():.2f} g/dL  (children 6-59 months, Ghana)")
    print(f"test : {ext}  n={len(df)}  mean Hb {hb_true.mean():.2f} g/dL  "
          f"(adults 19-88, {df['site'].value_counts().to_dict()})")
    print(f"population shift in the target itself: "
          f"{hb_true.mean() - hb_train.mean():+.2f} g/dL")

    cfg = load_config(run / "config.yaml")
    feats = colour_features_for(df, ext, cfg.roi_black_threshold,
                                out / "colour_features_external.csv")
    train_feats = pd.read_csv("reports/baselines/colour_features.csv")
    train_feats = train_feats.set_index("image_id").loc[train_df["image_id"]]
    train_feats.index = train_df.index

    results = {}
    # The floor: predicting the Ghanaian training mean for Italian/Indian adults.
    results["train_mean"] = report(
        "predict the CP-AnemiC training mean", hb_true,
        B.mean_hb_predictions(hb_train, len(df)), hb_train, cutoff,
        cfg.spec_target)
    # The colour baseline, fitted on Ghana only.
    ridge_pred, _ = B.fit_colour_ridge(train_feats[FEATURE_ORDER],
                                       hb_train, feats[FEATURE_ORDER])
    results["colour_ridge"] = report("colour ridge trained on Ghana", hb_true,
                                     ridge_pred, hb_train, cutoff, cfg.spec_target)
    try:
        results["cnn"] = report(f"ConvNeXt-Tiny ({run.name})", hb_true,
                                predict_cnn(run, df, ext), hb_train, cutoff,
                                cfg.spec_target)
    except ImportError as exc:
        print(f"\n  skipping the CNN (torch unavailable: {exc})")

    payload = {"run": str(run), "external": str(ext),
               "n_train": int(len(train_df)), "n_test": int(len(df)),
               "train_mean_hb": float(hb_train.mean()),
               "test_mean_hb": float(hb_true.mean()),
               "results": results}
    (out / "external_metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved -> {out / 'external_metrics.json'}")
    print("\nRead the raw MAE as the deployable number. The bias-corrected one "
          "uses test labels and cannot be claimed as performance.")


if __name__ == "__main__":
    main()
