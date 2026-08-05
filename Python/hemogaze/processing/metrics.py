"""Honest evaluation metrics for HemoGaze.

The whole point of this module: accuracy is banned as a headline number
(the dataset is imbalanced and accuracy hides the errors that matter in
screening). Everything here is built to answer the questions a clinician
and a skeptical reviewer would actually ask.

Nothing in here imports torch, so it can be unit-tested on synthetic data
with no GPU and no dataset present.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


# Above this, a result is treated as evidence of leakage or a confounder rather
# than as a success. See CLAUDE.md rule 4.
RED_FLAG_AUROC = 0.99

# Below this, a model is close enough to coin-flipping that derived comparisons
# (notably the generalisation gap) stop meaning anything.
NEAR_CHANCE_AUROC = 0.60

# Leave-one-site-out folds smaller than this are reported but kept out of the
# cross-site average: on a handful of images AUROC is dominated by which pair
# happened to swap, not by the model.
MIN_SITE_N = 30


@dataclass
class ClassificationReport:
    n: int
    prevalence: float          # fraction positive (anemic) in this set
    auroc: float
    auprc: float               # more honest than AUROC under class imbalance
    sens_at_spec: float        # sensitivity at a fixed specificity target
    spec_target: float
    threshold_at_spec: float
    ece: float                 # expected calibration error
    note: str = ""             # why a field is NaN, or which red flag tripped

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def is_red_flag(self) -> bool:
        """AUROC >= 0.99 is a leakage/confounder suspicion, not a trophy."""
        return np.isfinite(self.auroc) and self.auroc >= RED_FLAG_AUROC

    def summary_line(self) -> str:
        """One-line human summary. Deliberately puts AUPRC next to AUROC and
        never prints accuracy."""
        flag = "  [!] RED FLAG: suspect leakage/confounder" if self.is_red_flag else ""
        return (f"n={self.n:<5d} prev={self.prevalence:.3f}  "
                f"AUROC={self.auroc:.3f}  AUPRC={self.auprc:.3f}  "
                f"sens@spec{self.spec_target:.2f}={self.sens_at_spec:.3f}  "
                f"ECE={self.ece:.3f}{flag}")


def sensitivity_at_specificity(y_true, y_score, spec_target: float = 0.90):
    """Sensitivity (recall on positives) at the operating point that achieves
    at least `spec_target` specificity.

    In screening you fix the false-alarm rate you can tolerate and then ask
    how many true cases you catch. Reporting a single accuracy number instead
    of this is one of the most common ways anemia papers oversell.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    spec = 1.0 - fpr
    ok = spec >= spec_target
    if not ok.any():
        return 0.0, float("nan")
    # among thresholds meeting the specificity target, take the most sensitive
    idx_candidates = np.where(ok)[0]
    best = idx_candidates[np.argmax(tpr[idx_candidates])]
    return float(tpr[best]), float(thresholds[best])


def expected_calibration_error(y_true, y_prob, n_bins: int = 10):
    """Expected Calibration Error. Do the predicted probabilities mean what
    they say? A model can have great AUROC and still be badly calibrated;
    for a screening tool the probability is what a health worker trusts.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    """Return (mean_predicted, observed_fraction, count) per bin for plotting
    a reliability diagram. Empty bins are dropped."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if not mask.any():
            continue
        xs.append(float(y_prob[mask].mean()))
        ys.append(float(y_true[mask].mean()))
        ns.append(int(mask.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def classification_report(y_true, y_score, spec_target: float = 0.90,
                          n_bins: int = 10) -> ClassificationReport:
    """Full honest report. Never returns an accuracy.

    Single-class evaluation sets are returned as NaN with an explanatory note
    rather than raising: leave-one-site-out will hand us sites that are entirely
    non-anemic, and the correct response is to report "not evaluable here",
    not to crash the sweep or to invent a number.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) == 0:
        raise ValueError("Empty evaluation set.")

    if len(np.unique(y_true)) < 2:
        only = int(y_true[0])
        return ClassificationReport(
            n=int(len(y_true)), prevalence=float(y_true.mean()),
            auroc=float("nan"), auprc=float("nan"),
            sens_at_spec=float("nan"), spec_target=spec_target,
            threshold_at_spec=float("nan"),
            ece=expected_calibration_error(y_true, y_score, n_bins),
            note=(f"NOT EVALUABLE: every label in this set is {only}. "
                  f"Ranking metrics are undefined on one class -- this is "
                  f"itself a confounder finding, report it as such."),
        )

    sens, thr = sensitivity_at_specificity(y_true, y_score, spec_target)
    rep = ClassificationReport(
        n=int(len(y_true)),
        prevalence=float(y_true.mean()),
        auroc=float(roc_auc_score(y_true, y_score)),
        auprc=float(average_precision_score(y_true, y_score)),
        sens_at_spec=sens,
        spec_target=spec_target,
        threshold_at_spec=thr,
        ece=expected_calibration_error(y_true, y_score, n_bins),
    )
    if rep.is_red_flag:
        rep.note = (f"AUROC >= {RED_FLAG_AUROC} -- treat as suspected leakage "
                    f"or a site/camera confounder until proven otherwise.")
    return rep


# ---- The headline number: optimistic split vs unseen site -------------------

def generalisation_gap(patient_level: ClassificationReport,
                       site_out: list[ClassificationReport],
                       min_site_n: int = MIN_SITE_N) -> dict:
    """Quantify the finding this whole repo exists to measure.

    Two kinds of site are excluded from the average but still counted, because
    "how many sites were unusable" is part of the honest answer:

    * **single-class** sites, where ranking metrics are undefined;
    * **tiny** sites (n < ``min_site_n``). On CP-AnemiC two facilities
      contribute 8 and 15 images and score AUROC 0.86 and 1.00 — on 8 images a
      single swapped pair moves AUROC by ~0.1, so those folds are noise. Letting
      them into an unweighted mean pulled the cross-site average up by 0.07 and
      made the generalisation gap look like +0.01 when the evaluable sites say
      +0.08. Averaging them as equals is how a model gets credit for a fold that
      could not have measured anything.

    A large positive gap means the patient-level number was measuring the
    facility as much as the child.
    """
    usable = [r for r in site_out if np.isfinite(r.auroc)]
    n_single_class = len(site_out) - len(usable)
    evaluable = [r for r in usable if r.n >= min_site_n]
    n_too_small = len(usable) - len(evaluable)
    n_skipped = n_single_class + n_too_small
    if not evaluable:
        return {"patient_level_auroc": patient_level.auroc,
                "site_out_auroc_mean": float("nan"),
                "auroc_gap": float("nan"),
                "patient_level_auprc": patient_level.auprc,
                "site_out_auprc_mean": float("nan"),
                "auprc_gap": float("nan"),
                "n_sites_evaluated": 0, "n_sites_skipped": n_skipped,
                "n_sites_single_class": n_single_class,
                "n_sites_too_small": n_too_small, "min_site_n": min_site_n,
                "note": "No site was evaluable; cross-site claims are unsupported."}
    auroc_mean = float(np.mean([r.auroc for r in evaluable]))
    auprc_mean = float(np.mean([r.auprc for r in evaluable]))

    # A gap is only interpretable if there was something to lose. If the
    # in-distribution model is at chance, a small gap means "bad everywhere",
    # not "generalises well" -- and that misreading is an easy way to launder a
    # failed model into a reassuring sentence.
    note = ("Report both numbers together. The gap, not the higher number, "
            "is the result.")
    if patient_level.auroc < NEAR_CHANCE_AUROC:
        note = (f"NOT INTERPRETABLE AS GENERALISATION: patient-level AUROC "
                f"{patient_level.auroc:.3f} is at or near chance, so there was "
                f"no in-distribution performance to lose. A small gap here means "
                f"the model is weak everywhere, not that it transfers.")
    return {
        "patient_level_auroc": patient_level.auroc,
        "site_out_auroc_mean": auroc_mean,
        "site_out_auroc_min": float(np.min([r.auroc for r in evaluable])),
        "auroc_gap": float(patient_level.auroc - auroc_mean),
        "patient_level_auprc": patient_level.auprc,
        "site_out_auprc_mean": auprc_mean,
        "auprc_gap": float(patient_level.auprc - auprc_mean),
        "n_sites_evaluated": len(evaluable),
        "n_sites_skipped": n_skipped,
        "n_sites_single_class": n_single_class,
        "n_sites_too_small": n_too_small,
        "min_site_n": min_site_n,
        "note": note,
    }


# ---- Regression on hemoglobin (g/dL) ---------------------------------------

@dataclass
class RegressionReport:
    n: int
    mae: float                 # mean absolute error, g/dL
    rmse: float
    bias: float                # mean(pred - true): systematic over/under estimate
    loa_low: float             # Bland-Altman 95% limits of agreement
    loa_high: float
    mae_vs_trivial: float = float("nan")   # <1 beats predicting the train mean
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def beats_trivial(self) -> bool:
        return np.isfinite(self.mae_vs_trivial) and self.mae_vs_trivial < 0.98

    def summary_line(self) -> str:
        """MAE in g/dL is the headline because a clinician can read it; the
        ratio to the trivial baseline is next to it so nobody mistakes a
        respectable-looking error for a working model."""
        ratio = ("" if not np.isfinite(self.mae_vs_trivial)
                 else f"  ({self.mae_vs_trivial:.2f}x trivial"
                      f"{'' if self.beats_trivial else ' -- NO BETTER'})")
        return (f"n={self.n:<5d} MAE={self.mae:.2f} g/dL  RMSE={self.rmse:.2f}  "
                f"bias={self.bias:+.2f}  LoA=[{self.loa_low:+.2f}, "
                f"{self.loa_high:+.2f}]{ratio}")


def trivial_mae(y_train, y_true) -> float:
    """MAE of predicting the training mean for everyone -- the regression
    equivalent of the majority-class baseline, and the number every model here
    has to beat before it has demonstrated anything.

    On CP-AnemiC this is 1.80 g/dL. A model reporting 1.75 has not learned to
    read pallor; it has learned the average child.
    """
    y_train = np.asarray(y_train, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.abs(y_true - y_train.mean()).mean())


def regression_report(y_true, y_pred, y_train=None) -> RegressionReport:
    """For the Hb-regression variant. Bland-Altman (bias + limits of
    agreement) is the standard way to compare a new measurement against a
    reference in clinical work -- far more informative than R^2 alone.

    Pass ``y_train`` to get ``mae_vs_trivial``: the ratio of this model's MAE to
    the MAE of predicting the training mean. Below 1.0 the model beats the
    trivial baseline; at or above 1.0 it does not, no matter how respectable the
    absolute error looks in g/dL.

    Regression is preferred over binary classification here for a reason worth
    restating: hemoglobin is population-independent, so one model serves
    children (WHO cutoff 11 g/dL) and adults (12 for women, 13 for men) by
    thresholding afterwards. A binary model is married to the cutoff it was
    trained on.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        raise ValueError("Empty evaluation set.")
    diff = y_pred - y_true
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    mae = float(np.abs(diff).mean())

    ratio, note = float("nan"), ""
    if y_train is not None:
        triv = trivial_mae(y_train, y_true)
        ratio = mae / triv if triv > 0 else float("nan")
        if np.isfinite(ratio) and ratio >= 0.98:
            note = (f"MAE {mae:.2f} g/dL vs {triv:.2f} for predicting the "
                    f"training mean -- this model has not learned to read "
                    f"pallor. Report it as such.")
    return RegressionReport(
        n=int(len(y_true)),
        mae=mae,
        rmse=float(np.sqrt((diff ** 2).mean())),
        bias=bias,
        loa_low=bias - 1.96 * sd,
        loa_high=bias + 1.96 * sd,
        mae_vs_trivial=ratio,
        note=note,
    )


def classify_from_hb(hb_pred, cutoff) -> np.ndarray:
    """Turn predicted hemoglobin into anemia scores at any WHO cutoff.

    This is the whole argument for regressing Hb rather than classifying: the
    threshold is applied *after* the model, so one set of predictions serves
    children at 11 g/dL and adults at 12/13, and ``cutoff`` may be an array when
    it varies per person (sex-specific adult thresholds).

    Returns a score that increases with anemia risk, so it flows straight into
    ``classification_report`` alongside every other model in this repo.
    """
    return np.asarray(cutoff, dtype=float) - np.asarray(hb_pred, dtype=float)
