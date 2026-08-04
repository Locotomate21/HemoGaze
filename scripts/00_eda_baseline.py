#!/usr/bin/env python
"""Step 0: EDA, confounder audit, and the baselines. Run this BEFORE any CNN.

Three jobs, in the order that keeps the project honest:

  1. Audit the data -- class balance, per-site prevalence, confounder flags,
     and whether any site is so one-sided that it is not evaluable at all.
  2. Fit the baselines a deep model has to beat: majority class, colour
     logistic regression, and the site-prior confounder probe.
  3. Report them on BOTH a patient-level split and a leave-one-site-out sweep
     over *every* site, then print the generalisation gap. That gap is the
     headline result of the project, and it is available on day one, before a
     single GPU hour is spent.

Everything is written to ``reports/baselines/`` -- ``02_train.py`` refuses to
train until those files exist, which is how CLAUDE.md rule 2 is enforced in code
rather than in good intentions.

Usage:
    python scripts/00_eda_baseline.py --config config/default.yaml
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
from hemogaze import splits as S
from hemogaze.config import Config, load_config
from hemogaze.features import FEATURE_ORDER, color_features, roi_mask


def load_metadata(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.metadata_csv)
    if not path.exists():
        sys.exit(
            f"\nMetadata not found at {path}.\n"
            f"Either download CP-AnemiC (see data/README.md) into {cfg.data_dir} "
            f"with a metadata.csv holding columns {S.REQUIRED_COLUMNS} "
            f"(+ optional {S.OPTIONAL_COLUMNS}), or generate the synthetic "
            f"stand-in to test the pipeline:\n"
            f"    python scripts/make_synthetic_data.py --out data/synthetic\n"
            f"    python scripts/00_eda_baseline.py --config config/synthetic.yaml\n"
        )
    df = pd.read_csv(path)
    S._check_columns(df)
    return df


def build_feature_matrix(df: pd.DataFrame, cfg: Config, cache: Path) -> pd.DataFrame:
    """Colour features for every image, cached to disk.

    Cached because the leave-one-site-out sweep re-reads the same images once per
    site, and re-decoding every JPEG ten times is a waste that discourages people
    from running the full honest sweep.

    The returned frame carries ``df``'s own index, so split indices can be used
    with ``.loc`` on both. Positional alignment between two frames is exactly the
    kind of silent mismatch that would scramble labels against features.
    """
    def aligned(feats: pd.DataFrame) -> pd.DataFrame:
        out = feats.set_index("image_id").loc[df["image_id"]]
        out.index = df.index
        return out

    if cache.exists():
        cached = pd.read_csv(cache)
        if set(cached["image_id"]) == set(df["image_id"]):
            print(f"colour features: loaded cache {cache}")
            return aligned(cached)
        print("colour feature cache is stale; recomputing")

    from PIL import Image

    print(f"extracting colour features from {len(df)} images ...")
    # Statistics are taken over the ROI only. CP-AnemiC images are pre-segmented
    # conjunctiva strips on black, where the background is 55-87% of the frame
    # and varies 32 points between images -- unmasked, the features would encode
    # the crop outline instead of the pallor. On an unsegmented photo the mask
    # is ~all-True, so this stays correct either way.
    rows, bg_fracs = [], []
    for image_id in df["image_id"]:
        with Image.open(Path(cfg.data_dir) / image_id) as im:
            img = np.asarray(im.convert("RGB"))
        mask = roi_mask(img, cfg.roi_black_threshold)
        bg_fracs.append(1.0 - mask.mean())
        rows.append({"image_id": image_id, **color_features(img, mask)})
    feats = pd.DataFrame(rows)

    bg = np.asarray(bg_fracs)
    print(f"ROI masking: background is {bg.min():.0%}-{bg.max():.0%} of the "
          f"frame (median {np.median(bg):.0%}) and was excluded")
    if bg.max() < 0.02:
        print("  -> effectively no background found; images look unsegmented, "
              "so the whole frame is being used. Check that they are cropped "
              "to the conjunctiva.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(cache, index=False)
    print(f"colour features: cached -> {cache}")
    return aligned(feats)


def eda(df: pd.DataFrame) -> pd.DataFrame:
    print("\n===== EDA =====")
    print(f"images: {len(df)} | patients: {df['patient_id'].nunique()} | "
          f"sites: {df['site'].nunique()}")
    print(f"images per patient: "
          f"{df.groupby('patient_id').size().value_counts().to_dict()}")
    print(f"overall anemic prevalence: {df['label'].mean():.3f} "
          f"({int(df['label'].sum())} anemic / {len(df)} images)")
    if "hemoglobin" in df.columns:
        print(f"hemoglobin g/dL: mean={df['hemoglobin'].mean():.2f} "
              f"min={df['hemoglobin'].min():.1f} max={df['hemoglobin'].max():.1f}")

    balance = S.per_site_class_balance(df)
    print("\nper-site class balance (confounder_flag = >85% or <15% one class):")
    print(balance.to_string())

    flagged = balance.index[balance["confounder_flag"]].tolist()
    if flagged:
        print(f"\n[!] CONFOUNDER WARNING: site(s) {flagged} are near-single-class. "
              f"A model can score on those sites by recognising the facility "
              f"rather than the pallor. Read every cross-site number for them "
              f"with that in mind.")
    single = balance.index[(balance["prevalence_anemic"] == 0) |
                           (balance["prevalence_anemic"] == 1)].tolist()
    if single:
        print(f"[!] site(s) {single} contain exactly one class and cannot be "
              f"ranked at all; they are skipped in the AUROC average.")
    return balance


def evaluate_split(df: pd.DataFrame, feats: pd.DataFrame, split: S.Split,
                   cfg: Config) -> dict:
    """Fit and score all three baselines on one split."""
    tr, te = split.train_idx, split.test_idx
    X_tr = feats.loc[tr, FEATURE_ORDER].to_numpy(dtype=float)
    X_te = feats.loc[te, FEATURE_ORDER].to_numpy(dtype=float)
    y_tr = df.loc[tr, "label"].to_numpy().astype(int)
    y_te = df.loc[te, "label"].to_numpy().astype(int)

    scores = {
        "majority_class": B.majority_class_scores(y_tr, len(y_te)),
        "colour_logistic": B.fit_colour_logistic(X_tr, y_tr, X_te,
                                                 seed=cfg.seed)[0],
        "site_prior": B.site_prior_scores(df, tr, te),
    }
    reports = {name: M.classification_report(y_te, s, cfg.spec_target,
                                             cfg.n_calib_bins)
               for name, s in scores.items()}

    header = f"--- {split.kind} | {split.note} | split_hash={split.hash()}"
    print(f"\n{header}")
    print(f"    train n={len(tr)} (prev {y_tr.mean():.3f})   "
          f"test n={len(te)} (prev {y_te.mean():.3f})")
    for name, rep in reports.items():
        print(f"    {name:<16} {rep.summary_line()}")
        if rep.note:
            print(f"    {'':<16} -> {rep.note}")
    out = {
        "split_kind": split.kind, "split_note": split.note,
        "split_hash": split.hash(),
        "baselines": {name: rep.as_dict() for name, rep in reports.items()},
    }

    # The Hb-regression task, reported on the same split so the two views of the
    # same data sit side by side. Hemoglobin is population-independent: the WHO
    # cutoff is applied after the model, which is why this generalises to adult
    # datasets that a binary model trained at 11 g/dL cannot serve.
    if "hemoglobin" in df.columns:
        hb_tr = df.loc[tr, "hemoglobin"].to_numpy(dtype=float)
        hb_te = df.loc[te, "hemoglobin"].to_numpy(dtype=float)
        preds = {
            "mean_hb": B.mean_hb_predictions(hb_tr, len(hb_te)),
            "colour_ridge": B.fit_colour_ridge(X_tr, hb_tr, X_te)[0],
            "site_mean_hb": B.site_mean_hb(df, tr, te),
        }
        rregs = {n: M.regression_report(hb_te, p, y_train=hb_tr)
                 for n, p in preds.items()}
        print("    -- hemoglobin regression (g/dL) --")
        for name, rep in rregs.items():
            print(f"    {name:<16} {rep.summary_line()}")
            if rep.note:
                print(f"    {'':<16} -> {rep.note}")
        out["regression"] = {n: r.as_dict() for n, r in rregs.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    df = load_metadata(cfg)
    tag = cfg.data_tag()
    if cfg.is_synthetic():
        print("\n" + "=" * 72)
        print("SYNTHETIC DATA. These numbers test the pipeline. They are not")
        print("results about anemia and must never be quoted as such.")
        print("=" * 72)

    balance = eda(df)

    out = Path(cfg.baseline_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats = build_feature_matrix(df, cfg, out / "colour_features.csv")

    print("\n===== BASELINES =====")
    print("(the bar every deep-model result must clear before it is quoted)")

    patient_split = S.patient_level_split(df, cfg.val_frac, cfg.test_frac, cfg.seed)
    patient_res = evaluate_split(df, feats, patient_split, cfg)

    site_results, site_reports = [], []
    for site in sorted(df["site"].unique()):
        split = S.leave_one_site_out(df, site, cfg.val_frac, cfg.seed)
        res = evaluate_split(df, feats, split, cfg)
        res["test_site"] = site
        site_results.append(res)
        site_reports.append(M.ClassificationReport(
            **res["baselines"]["colour_logistic"]))

    patient_report = M.ClassificationReport(**patient_res["baselines"]["colour_logistic"])
    gap = M.generalisation_gap(patient_report, site_reports)

    print("\n===== THE HEADLINE: colour baseline, optimistic vs unseen site =====")
    print(f"    patient-level AUROC     {gap['patient_level_auroc']:.3f}")
    print(f"    leave-one-site-out mean {gap['site_out_auroc_mean']:.3f} "
          f"(worst site {gap.get('site_out_auroc_min', float('nan')):.3f}, "
          f"{gap['n_sites_evaluated']} sites evaluated; "
          f"{gap['n_sites_too_small']} excluded as smaller than "
          f"n={gap['min_site_n']}, {gap['n_sites_single_class']} as single-class)")
    print(f"    AUROC gap               {gap['auroc_gap']:+.3f}   <-- the finding")
    print(f"    AUPRC gap               {gap['auprc_gap']:+.3f}")
    print(f"    {gap['note']}")

    payload = {
        "data": tag,
        "synthetic": cfg.is_synthetic(),
        "seed": cfg.seed,
        "n_images": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "n_sites": int(df["site"].nunique()),
        "prevalence": float(df["label"].mean()),
        "per_site_balance": balance.reset_index().to_dict(orient="records"),
        "patient_level": patient_res,
        "leave_one_site_out": site_results,
        "generalisation_gap_colour_logistic": gap,
    }
    (out / "baseline_metrics.json").write_text(json.dumps(payload, indent=2))
    cfg.save(out / "config.yaml")
    print(f"\nsaved -> {out / 'baseline_metrics.json'}  (and the resolved config)")
    print("\nOnly now is it fair to train a CNN:  python scripts/02_train.py")


if __name__ == "__main__":
    main()
