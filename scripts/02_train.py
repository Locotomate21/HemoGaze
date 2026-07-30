#!/usr/bin/env python
"""Step 2: transfer-learning training. Only runs after 00_eda_baseline.py.

Deliberately minimal and readable -- this is a rigor project, not a framework.
What it does enforce:

* **The baseline gate.** It refuses to start unless ``reports/baselines/
  baseline_metrics.json`` exists, and it prints the colour-baseline number for
  the matching split next to every CNN number it produces. You cannot get a
  standalone deep-model result out of this script.
* **Bookkeeping.** Seed, resolved config, split hash, backbone, trainable
  parameter count and the data tag (real vs SYNTHETIC) are written next to the
  metrics for every run.
* **Model selection on AUPRC, never accuracy**, and a red-flag warning on
  AUROC >= 0.99.

Usage:
    python scripts/02_train.py --config config/default.yaml
    python scripts/02_train.py --config config/default.yaml --site-out SITE_A
    python scripts/02_train.py --config config/default.yaml --all-sites
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemogaze import metrics as M
from hemogaze import splits as S
from hemogaze.config import Config, load_config


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_baselines(cfg: Config) -> dict:
    """CLAUDE.md rule 2, enforced rather than requested."""
    path = Path(cfg.baseline_dir) / "baseline_metrics.json"
    if not path.exists():
        sys.exit(
            f"\nRefusing to train: no baselines at {path}.\n"
            f"A majority-class and colour-logistic baseline must exist and be "
            f"reported before any CNN number means anything. Run:\n"
            f"    python scripts/00_eda_baseline.py --config <your config>\n"
        )
    return json.loads(path.read_text())


def baseline_for(baselines: dict, site_out: str | None) -> dict:
    """The colour-baseline report for the same split this run uses, so the CNN
    is always quoted against a like-for-like comparison."""
    if site_out is None:
        return baselines["patient_level"]["baselines"]["colour_logistic"]
    for entry in baselines["leave_one_site_out"]:
        if entry["test_site"] == site_out:
            return entry["baselines"]["colour_logistic"]
    sys.exit(f"No baseline recorded for site {site_out!r}; rerun "
             f"scripts/00_eda_baseline.py so the comparison exists.")


def infer(model, dl, device):
    import torch
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x.to(device)).squeeze(1)
            ps.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_one(cfg: Config, df: pd.DataFrame, split: S.Split, baselines: dict,
              site_out: str | None, run_name: str) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from hemogaze.dataset import make_dataset
    from hemogaze.model import (build_model, freeze_backbone,
                                trainable_parameter_count)

    seed_everything(cfg.seed)
    device = cfg.resolve_device()

    def loader(idx, train):
        ds = make_dataset(df.loc[idx], cfg.data_dir, cfg.image_size, train)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train,
                          num_workers=cfg.num_workers)

    dl_tr, dl_val, dl_te = (loader(split.train_idx, True),
                            loader(split.val_idx, False),
                            loader(split.test_idx, False))

    model = build_model(cfg.backbone, cfg.pretrained, cfg.dropout).to(device)

    # Class imbalance -> weight the positive class, because the error that
    # matters in screening is a missed anemic child, not a false alarm.
    pos = float(df.loc[split.train_idx, "label"].mean())
    pos_weight = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    print(f"\n=== training {run_name} on {device} "
          f"({split.kind}: {split.note}, split_hash={split.hash()}) ===")
    print(f"    train {len(split.train_idx)} / val {len(split.val_idx)} / "
          f"test {len(split.test_idx)} images, train prevalence {pos:.3f}")

    best_auprc, best_state, patience = -1.0, None, 0
    frozen = False
    for epoch in range(cfg.epochs):
        # Head-only warmup: let the fresh head settle before the pretrained
        # features are allowed to move onto a few hundred images.
        want_frozen = epoch < cfg.warmup_epochs
        if want_frozen != frozen:
            freeze_backbone(model, want_frozen)
            frozen = want_frozen
            print(f"    epoch {epoch:02d}: backbone "
                  f"{'frozen (head warmup)' if frozen else 'unfrozen'}, "
                  f"{trainable_parameter_count(model):,} trainable params")

        model.train()
        running = 0.0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
            running += loss.detach().item() * len(y)

        # Select on AUPRC: imbalance-aware, and never accuracy.
        ys, ps = infer(model, dl_val, device)
        rep = M.classification_report(ys, ps, cfg.spec_target, cfg.n_calib_bins)
        print(f"    epoch {epoch:02d}  loss={running / len(split.train_idx):.4f}  "
              f"val_auprc={rep.auprc:.3f}  val_auroc={rep.auroc:.3f}  "
              f"val_ece={rep.ece:.3f}")

        if np.isfinite(rep.auprc) and rep.auprc > best_auprc:
            best_auprc = rep.auprc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"    early stopping at epoch {epoch}")
                break

    if best_state is None:
        sys.exit("Validation AUPRC was never computable (single-class val set?). "
                 "Fix the split before trusting anything from this run.")
    model.load_state_dict(best_state)

    ys, ps = infer(model, dl_te, device)
    test_rep = M.classification_report(ys, ps, cfg.spec_target, cfg.n_calib_bins)
    base = baseline_for(baselines, site_out)

    out = Path(cfg.out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.yaml")
    torch.save(best_state, out / "model.pt")
    np.savez(out / "test_preds.npz", y_true=ys, y_score=ps)

    record = {
        "run_name": run_name,
        "data": cfg.data_tag(),
        "synthetic": cfg.is_synthetic(),
        "seed": cfg.seed,
        "backbone": cfg.backbone,
        "pretrained": cfg.pretrained,
        "trainable_params": trainable_parameter_count(model),
        "device": device,
        "split_kind": split.kind,
        "split_note": split.note,
        "split_hash": split.hash(),
        "test_site": site_out,
        "cnn": test_rep.as_dict(),
        "colour_logistic_baseline": base,
        "cnn_minus_baseline_auroc": (test_rep.auroc - base["auroc"]
                                     if np.isfinite(test_rep.auroc) and
                                     np.isfinite(base["auroc"]) else None),
        "cnn_minus_baseline_auprc": (test_rep.auprc - base["auprc"]
                                     if np.isfinite(test_rep.auprc) and
                                     np.isfinite(base["auprc"]) else None),
    }
    (out / "test_metrics.json").write_text(json.dumps(record, indent=2))

    print(f"\n    TEST  cnn             {test_rep.summary_line()}")
    print(f"    TEST  colour baseline "
          f"AUROC={base['auroc']:.3f}  AUPRC={base['auprc']:.3f}  "
          f"sens@spec={base['sens_at_spec']:.3f}  ECE={base['ece']:.3f}")
    delta = record["cnn_minus_baseline_auroc"]
    if delta is not None:
        verdict = ("the CNN does NOT clearly beat colour logistic regression -- "
                   "report that plainly" if delta < 0.03
                   else "the CNN beats the colour baseline on this split")
        print(f"    DELTA AUROC {delta:+.3f}  -> {verdict}")
    if test_rep.note:
        print(f"    [!] {test_rep.note}")
    print(f"    saved -> {out}")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--site-out", default=None,
                    help="hold this site out for test; else patient-level split")
    ap.add_argument("--all-sites", action="store_true",
                    help="loop leave-one-site-out over every site")
    args = ap.parse_args()
    if args.site_out and args.all_sites:
        sys.exit("Use --site-out or --all-sites, not both.")

    cfg = load_config(args.config)
    baselines = load_baselines(cfg)
    df = pd.read_csv(cfg.metadata_csv)
    S._check_columns(df)

    if cfg.is_synthetic():
        print("[SYNTHETIC DATA] this run exercises the pipeline; its numbers are "
              "not results about anemia.")

    if args.all_sites:
        records = []
        for site in sorted(df["site"].unique()):
            split = S.leave_one_site_out(df, site, cfg.val_frac, cfg.seed)
            records.append(train_one(cfg, df, split, baselines, site,
                                     f"{cfg.run_name}_siteout_{site}"))
        summary = Path(cfg.out_dir) / f"{cfg.run_name}_all_sites.json"
        summary.write_text(json.dumps(records, indent=2))
        aurocs = [r["cnn"]["auroc"] for r in records
                  if np.isfinite(r["cnn"]["auroc"])]
        print(f"\n===== leave-one-site-out sweep, {len(aurocs)} evaluable sites =====")
        print(f"    CNN AUROC mean {np.mean(aurocs):.3f}  "
              f"min {np.min(aurocs):.3f}  max {np.max(aurocs):.3f}")
        print(f"    saved -> {summary}")
        print("    Now compare against the patient-level run. The gap is the result.")
        return

    split = (S.leave_one_site_out(df, args.site_out, cfg.val_frac, cfg.seed)
             if args.site_out else
             S.patient_level_split(df, cfg.val_frac, cfg.test_frac, cfg.seed))
    run_name = (f"{cfg.run_name}_siteout_{args.site_out}" if args.site_out
                else f"{cfg.run_name}_patient_level")
    train_one(cfg, df, split, baselines, args.site_out, run_name)


if __name__ == "__main__":
    main()
