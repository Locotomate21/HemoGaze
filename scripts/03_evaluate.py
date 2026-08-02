#!/usr/bin/env python
"""Step 3: turn saved predictions into the figures and the table that make the
case -- including the parts that make the case *against* the model.

Per run (``--run reports/runs/<name>``):
  * the honest report printed in full, CNN next to the colour baseline;
  * a reliability diagram, because a screening probability a health worker reads
    off a phone has to mean what it says;
  * a score-distribution plot by true class;
  * Grad-CAM overlays, to check the model looks at the conjunctiva and not at a
    flash reflection, an eyelash, or something only one facility had in frame.

Across runs (``--compare``):
  * a markdown table of patient-level vs every left-out site, with the
    generalisation gap, written to ``reports/summary.md``.

Usage:
    python scripts/03_evaluate.py --run reports/runs/baseline_patient_level
    python scripts/03_evaluate.py --compare
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemogaze import metrics as M
from hemogaze.config import load_config


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def reliability_plot(y_true, y_score, out_path: Path, n_bins: int = 10) -> None:
    plt = _plt()
    xs, ys, ns = M.reliability_curve(y_true, y_score, n_bins)
    ece = M.expected_calibration_error(y_true, y_score, n_bins)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    ax.plot(xs, ys, "o-", label="model")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(4, -9),
                    fontsize=7, color="dimgray")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability of anemia")
    ax.set_ylabel("observed fraction anemic")
    ax.set_title(f"Reliability (ECE={ece:.3f})\nbin counts annotated", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    saved {out_path}")


def score_distribution_plot(y_true, y_score, out_path: Path) -> None:
    """Where the two classes actually sit. A single overlapping blob explains a
    mediocre AUROC far better than any summary statistic."""
    plt = _plt()
    y_true = np.asarray(y_true).astype(int)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    bins = np.linspace(0, 1, 21)
    ax.hist(y_score[y_true == 0], bins=bins, alpha=0.6, label="not anemic")
    ax.hist(y_score[y_true == 1], bins=bins, alpha=0.6, label="anemic")
    ax.set_xlabel("predicted probability of anemia")
    ax.set_ylabel("images")
    ax.set_title("Score separation by true label", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    saved {out_path}")


def gradcam_grid(run: Path, meta: dict, n_images: int = 8) -> None:
    """Grad-CAM, implemented with plain hooks so the repo needs no extra
    dependency for its most important sanity check.

    Read the output adversarially: a heatmap centred on the conjunctiva supports
    the model; one sitting on the eyelid margin, a specular highlight, or the
    image border says the model found a shortcut.
    """
    weights = run / "model.pt"
    if not weights.exists():
        print("    no model.pt in this run; skipping Grad-CAM")
        return
    try:
        import torch
        from hemogaze.dataset import IMAGENET_MEAN, IMAGENET_STD, make_dataset
        from hemogaze.model import build_model, gradcam_target_layer, load_weights
    except ImportError as exc:
        print(f"    skipping Grad-CAM (torch stack unavailable: {exc})")
        return

    cfg = load_config(run / "config.yaml")
    df = pd.read_csv(cfg.metadata_csv)
    # Show the test images of this run: the ones the model never trained on.
    from hemogaze import splits as S
    split = (S.leave_one_site_out(df, meta["test_site"], cfg.val_frac, cfg.seed)
             if meta.get("test_site") else
             S.patient_level_split(df, cfg.val_frac, cfg.test_frac, cfg.seed))
    test_df = df.loc[split.test_idx]
    # A balanced handful, so the panel is not all one class.
    picks = pd.concat([test_df[test_df["label"] == c].head(n_images // 2)
                       for c in (1, 0)])
    if picks.empty:
        print("    no test images to visualise; skipping Grad-CAM")
        return

    device = "cpu"   # a handful of images; keep it simple and deterministic
    model = build_model(cfg.backbone, pretrained=False, dropout=cfg.dropout)
    load_weights(model, weights, device).to(device).eval()
    target = gradcam_target_layer(model)

    acts, grads = {}, {}
    h1 = target.register_forward_hook(
        lambda m, i, o: acts.__setitem__("a", o.detach()))
    h2 = target.register_full_backward_hook(
        lambda m, gi, go: grads.__setitem__("g", go[0].detach()))

    # Must mirror the run's own evaluation preprocessing. A model tested on a
    # grey-filled background but visualised on the raw black one would produce a
    # heatmap of a situation it was never scored in.
    ds = make_dataset(picks, cfg.data_dir, cfg.image_size, train=False,
                      crop_to_roi=cfg.crop_to_roi,
                      randomise_background=cfg.randomise_background,
                      seed=cfg.seed)
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)

    plt = _plt()
    cols = min(len(ds), n_images)
    fig, axes = plt.subplots(2, cols, figsize=(1.9 * cols, 4.2), squeeze=False)
    for j in range(cols):
        x, y = ds[j]
        logit = model(x.unsqueeze(0).to(device)).squeeze()
        model.zero_grad()
        logit.backward()

        a, g = acts["a"][0], grads["g"][0]                 # (C, h, w)
        w = g.mean(dim=(1, 2), keepdim=True)               # channel importance
        cam = torch.relu((w * a).sum(0))
        cam = cam / (cam.max() + 1e-8)
        cam = torch.nn.functional.interpolate(
            cam[None, None], size=x.shape[-2:], mode="bilinear",
            align_corners=False)[0, 0].numpy()

        img = np.clip(x.numpy() * std + mean, 0, 1).transpose(1, 2, 0)
        prob = torch.sigmoid(logit.detach()).item()
        axes[0][j].imshow(img); axes[0][j].axis("off")
        axes[0][j].set_title(f"true={int(y)}  p={prob:.2f}", fontsize=8)
        axes[1][j].imshow(img); axes[1][j].imshow(cam, cmap="jet", alpha=0.45)
        axes[1][j].axis("off")
    fig.suptitle("Grad-CAM on held-out images -- is it looking at the conjunctiva?",
                 fontsize=10)
    fig.tight_layout()
    out = run / "gradcam.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    h1.remove(); h2.remove()
    print(f"    saved {out}")


def evaluate_run(run: Path, n_bins: int = 10) -> None:
    preds = run / "test_preds.npz"
    if not preds.exists():
        sys.exit(f"No test_preds.npz in {run}. Train first (scripts/02_train.py).")
    d = np.load(preds)
    y_true, y_score = d["y_true"], d["y_score"]
    meta = {}
    if (run / "test_metrics.json").exists():
        meta = json.loads((run / "test_metrics.json").read_text())

    print(f"\n===== {run.name} =====")
    if meta.get("synthetic"):
        print("[SYNTHETIC DATA] pipeline test only, not a result about anemia.")
    if meta:
        print(f"split: {meta['split_kind']} ({meta['split_note']})  "
              f"split_hash={meta['split_hash']}  backbone={meta['backbone']}")

    rep = M.classification_report(y_true, y_score, n_bins=n_bins)
    print("\nhonest test report (no accuracy, by design):")
    for k, v in rep.as_dict().items():
        print(f"    {k}: {v}")
    if meta.get("colour_logistic_baseline"):
        base = meta["colour_logistic_baseline"]
        print(f"\ncolour logistic baseline on the same split: "
              f"AUROC={base['auroc']:.3f}  AUPRC={base['auprc']:.3f}")
        print(f"CNN minus baseline AUROC: {meta['cnn_minus_baseline_auroc']:+.3f}")
    if rep.is_red_flag:
        print("\n[!] AUROC >= 0.99. Treat as suspected leakage or a site/camera "
              "confounder. Run the data-integrity agent before believing it.")

    print("\nfigures:")
    reliability_plot(y_true, y_score, run / "reliability.png", n_bins)
    score_distribution_plot(y_true, y_score, run / "score_distribution.png")
    gradcam_grid(run, meta)


def compare(runs_dir: Path, baseline_json: Path, out_md: Path) -> None:
    """Assemble the one table this project exists to produce."""
    records = []
    for path in sorted(runs_dir.glob("*/test_metrics.json")):
        records.append(json.loads(path.read_text()))
    if not records:
        sys.exit(f"No runs with test_metrics.json under {runs_dir}.")

    rows = []
    for r in records:
        rows.append({
            "run": r["run_name"],
            "split": r["split_kind"],
            "test_site": r.get("test_site") or "-",
            "n_test": r["cnn"]["n"],
            "prevalence": round(r["cnn"]["prevalence"], 3),
            "cnn_auroc": round(r["cnn"]["auroc"], 3),
            "cnn_auprc": round(r["cnn"]["auprc"], 3),
            "cnn_sens@spec": round(r["cnn"]["sens_at_spec"], 3),
            "cnn_ece": round(r["cnn"]["ece"], 3),
            "colour_auroc": round(r["colour_logistic_baseline"]["auroc"], 3),
            "delta_auroc": (None if r["cnn_minus_baseline_auroc"] is None
                            else round(r["cnn_minus_baseline_auroc"], 3)),
        })
    table = pd.DataFrame(rows)

    lines = ["# HemoGaze results summary", ""]
    if any(r.get("synthetic") for r in records):
        lines += ["> **SYNTHETIC DATA.** These numbers exercise the pipeline. "
                  "They are not findings about anemia.", ""]
    try:
        rendered = table.to_markdown(index=False)
    except ImportError:      # pandas needs `tabulate` for markdown output
        rendered = "```\n" + table.to_string(index=False) + "\n```"
    lines += ["Accuracy is deliberately absent. `sens@spec` is sensitivity at "
              "the configured specificity target; `ece` is expected calibration "
              "error (lower is better).", "", rendered, ""]

    patient = [r for r in records if r["split_kind"] == "patient_level"]
    # Pass every site fold through, including the unevaluable ones: the gap
    # function is what decides which folds count, and it reports how many it
    # had to drop. Filtering them out here would hide that from the summary.
    site = [r for r in records if r["split_kind"] == "leave_one_site_out"]
    if patient and site:
        p = M.ClassificationReport(**patient[0]["cnn"])
        gap = M.generalisation_gap(p, [M.ClassificationReport(**r["cnn"])
                                       for r in site])
        worst = gap.get("site_out_auroc_min", float("nan"))
        lines += ["## The headline: generalisation gap (CNN)", "",
                  f"- patient-level AUROC: **{gap['patient_level_auroc']:.3f}**",
                  f"- leave-one-site-out mean AUROC: "
                  f"**{gap['site_out_auroc_mean']:.3f}** "
                  f"(worst site {worst:.3f}; "
                  f"{gap['n_sites_evaluated']} sites evaluated, "
                  f"{gap['n_sites_too_small']} excluded as smaller than "
                  f"n={gap['min_site_n']}, "
                  f"{gap['n_sites_single_class']} as single-class)",
                  f"- **AUROC gap: {gap['auroc_gap']:+.3f}** — this, not the "
                  f"higher number, is the result.",
                  f"- AUPRC gap: {gap['auprc_gap']:+.3f}",
                  "", f"> {gap['note']}", ""]
    elif patient or site:
        lines += ["## Generalisation gap", "",
                  "_Not computable yet: run both a patient-level and a "
                  "`--all-sites` training pass before quoting a gap._", ""]

    if baseline_json.exists():
        b = json.loads(baseline_json.read_text())
        bg = b["generalisation_gap_colour_logistic"]
        lines += ["## Colour-baseline reference (from step 0)", "",
                  f"- patient-level AUROC {bg['patient_level_auroc']:.3f}, "
                  f"leave-one-site-out mean {bg['site_out_auroc_mean']:.3f}, "
                  f"gap {bg['auroc_gap']:+.3f}",
                  f"- data: {b['n_images']} images, {b['n_patients']} patients, "
                  f"{b['n_sites']} sites, prevalence {b['prevalence']:.3f}", ""]

    lines += ["## Scope", "",
              "Screening signal, not a diagnosis; lab hemoglobin is the "
              "reference standard. Population is the dataset's own: no claim "
              "extends beyond it.", ""]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nsaved -> {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="reports/runs/<name> dir")
    ap.add_argument("--compare", action="store_true",
                    help="build reports/summary.md across all runs")
    ap.add_argument("--runs-dir", default="reports/runs")
    ap.add_argument("--baselines", default="reports/baselines/baseline_metrics.json")
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()
    if not args.run and not args.compare:
        ap.error("pass --run <dir> and/or --compare")

    if args.run:
        evaluate_run(Path(args.run), args.n_bins)
    if args.compare:
        compare(Path(args.runs_dir), Path(args.baselines),
                Path("reports/summary.md"))


if __name__ == "__main__":
    main()
