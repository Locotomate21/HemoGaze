"""Data splitting -- the single most important file in this repo.

Two split strategies, and you should ALWAYS report both:

1. patient_level_split: random split, but the unit is the patient, never the
   image. Two eyes / two photos of the same child must never land on opposite
   sides. This gives an optimistic "in-distribution" estimate.

2. leave_one_site_out: hold out entire collection sites. CP-AnemiC was
   gathered across ten health facilities in Ghana, and each image is tagged
   with its site. If a model does great on a random split but collapses when
   evaluated on an unseen site, it learned the camera / lighting / facility,
   not the pallor. The gap between (1) and (2) is the headline result of the
   whole project.

There is also an explicit leakage assertion you can call from tests and from
the data-integrity agent.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["image_id", "patient_id", "site", "label"]  # label: 1=anemic
OPTIONAL_COLUMNS = ["hemoglobin", "age_months", "sex"]


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    kind: str
    note: str = ""

    def hash(self) -> str:
        """Stable fingerprint of the split, so every experiment can record
        exactly which split produced its numbers."""
        h = hashlib.sha256()
        for name in ("train_idx", "val_idx", "test_idx"):
            arr = np.sort(getattr(self, name)).astype(np.int64).tobytes()
            h.update(name.encode())
            h.update(arr)
        return h.hexdigest()[:12]


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Metadata is missing required columns {missing}. "
            f"Without patient_id and site you cannot do honest evaluation -- "
            f"stop and fix the metadata before training."
        )


def patient_level_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.15,
                        seed: int = 42) -> Split:
    _check_columns(df)
    rng = np.random.default_rng(seed)
    patients = np.asarray(df["patient_id"].unique(), dtype=object)
    rng.shuffle(patients)
    n = len(patients)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_p = set(patients[:n_test])
    val_p = set(patients[n_test:n_test + n_val])
    train_p = set(patients[n_test + n_val:])

    idx = df.index.to_numpy()
    train_idx = idx[df["patient_id"].isin(train_p).to_numpy()]
    val_idx = idx[df["patient_id"].isin(val_p).to_numpy()]
    test_idx = idx[df["patient_id"].isin(test_p).to_numpy()]
    s = Split(train_idx, val_idx, test_idx, kind="patient_level",
              note=f"seed={seed}")
    assert_no_leakage(df, s)
    return s


def leave_one_site_out(df: pd.DataFrame, test_site, val_frac=0.15,
                       seed: int = 42) -> Split:
    """Hold out one entire site for test; carve a patient-level val set from
    the remaining sites. Iterate this over every site for the real evaluation.
    """
    _check_columns(df)
    if test_site not in set(df["site"].unique()):
        raise ValueError(f"Unknown site {test_site!r}. "
                         f"Available: {sorted(df['site'].unique())}")
    idx = df.index.to_numpy()
    test_mask = (df["site"] == test_site).to_numpy()
    test_idx = idx[test_mask]

    rest = df[~test_mask]
    rng = np.random.default_rng(seed)
    patients = np.asarray(rest["patient_id"].unique(), dtype=object)
    rng.shuffle(patients)
    n_val = int(round(len(patients) * val_frac))
    val_p = set(patients[:n_val])
    val_idx = rest.index.to_numpy()[rest["patient_id"].isin(val_p).to_numpy()]
    train_idx = rest.index.to_numpy()[~rest["patient_id"].isin(val_p).to_numpy()]
    s = Split(train_idx, val_idx, test_idx, kind="leave_one_site_out",
              note=f"test_site={test_site}, seed={seed}")
    assert_no_leakage(df, s)
    return s


def assert_no_leakage(df: pd.DataFrame, s: Split) -> None:
    """Hard guard. Raises if a patient appears on more than one side, or (for
    site splits) if the held-out site bleeds into train/val. The data-integrity
    agent runs this before it will approve any training."""
    parts = {"train": s.train_idx, "val": s.val_idx, "test": s.test_idx}
    pats = {k: set(df.loc[v, "patient_id"]) for k, v in parts.items()}
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                overlap = pats[a] & pats[b]
                if overlap:
                    raise AssertionError(
                        f"PATIENT LEAKAGE between {a} and {b}: "
                        f"{len(overlap)} shared patient(s), e.g. {list(overlap)[:3]}"
                    )
    if s.kind == "leave_one_site_out":
        test_sites = set(df.loc[s.test_idx, "site"])
        train_val_sites = set(df.loc[np.concatenate([s.train_idx, s.val_idx]), "site"])
        bleed = test_sites & train_val_sites
        if bleed:
            raise AssertionError(f"SITE LEAKAGE: held-out site(s) {bleed} "
                                 f"also appear in train/val.")


def per_site_class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Confounder audit: if one site is almost entirely anemic and another
    almost entirely not, a model can score well by learning the site. Print
    this in EDA and look hard at any site that is >~85% one class."""
    _check_columns(df)
    g = df.groupby("site")["label"]
    out = pd.DataFrame({
        "n": g.size(),
        "prevalence_anemic": g.mean(),
    })
    out["confounder_flag"] = (out["prevalence_anemic"] > 0.85) | \
                             (out["prevalence_anemic"] < 0.15)
    return out.sort_values("prevalence_anemic", ascending=False)
