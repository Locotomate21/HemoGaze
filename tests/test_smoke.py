"""Smoke tests that run with no dataset and no GPU. They verify the parts of
the repo that MUST be correct: the leakage guards, the honest metrics, and the
baselines a deep model has to beat.

    pytest -q      (or)      python tests/test_smoke.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemogaze import baselines as B
from hemogaze import metrics as M
from hemogaze import splits as S
from hemogaze.features import (FEATURE_ORDER, color_features,
                               features_to_vector, roi_mask)


def _fake_meta(n_patients=60, sites=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        site = f"SITE_{p % sites}"
        label = int(rng.random() < 0.4)
        for eye in range(2):  # two images per patient
            rows.append(dict(image_id=f"p{p}_e{eye}.jpg", patient_id=f"p{p}",
                             site=site, label=label))
    return pd.DataFrame(rows)


# ---- splits and leakage -----------------------------------------------------

def test_patient_level_no_leakage():
    df = _fake_meta()
    s = S.patient_level_split(df)
    # the guard inside the function would already raise; assert disjoint too
    tr = set(df.loc[s.train_idx, "patient_id"])
    te = set(df.loc[s.test_idx, "patient_id"])
    assert tr.isdisjoint(te)


def test_patient_level_split_covers_every_row_once():
    """No row may be dropped or duplicated -- a silently shrinking test set is
    a very quiet way to report a better number than you earned."""
    df = _fake_meta()
    s = S.patient_level_split(df)
    allocated = np.concatenate([s.train_idx, s.val_idx, s.test_idx])
    assert sorted(allocated) == sorted(df.index.to_numpy())


def test_both_eyes_of_a_patient_stay_together():
    df = _fake_meta()
    s = S.patient_level_split(df)
    where = {}
    for name, idx in (("train", s.train_idx), ("val", s.val_idx),
                      ("test", s.test_idx)):
        for pid in df.loc[idx, "patient_id"]:
            where.setdefault(pid, set()).add(name)
    assert all(len(v) == 1 for v in where.values())


def test_leave_one_site_out_holds_site():
    df = _fake_meta()
    s = S.leave_one_site_out(df, "SITE_0")
    assert set(df.loc[s.test_idx, "site"]) == {"SITE_0"}
    assert "SITE_0" not in set(df.loc[s.train_idx, "site"])
    assert "SITE_0" not in set(df.loc[s.val_idx, "site"])


def test_leakage_guard_trips():
    df = _fake_meta()
    bad = S.Split(train_idx=df.index.to_numpy()[:10],
                  val_idx=np.array([], dtype=int),
                  test_idx=df.index.to_numpy()[:10],  # same rows in train+test
                  kind="patient_level")
    with pytest.raises(AssertionError):
        S.assert_no_leakage(df, bad)


def test_site_leakage_guard_trips():
    """A split labelled leave_one_site_out whose held-out site also appears in
    train must be rejected, even when no patient is shared."""
    df = _fake_meta()
    site0 = df.index[df["site"] == "SITE_0"].to_numpy()
    bad = S.Split(train_idx=site0[:20], val_idx=np.array([], dtype=int),
                  test_idx=site0[20:], kind="leave_one_site_out")
    with pytest.raises(AssertionError, match="SITE LEAKAGE"):
        S.assert_no_leakage(df, bad)


def test_missing_columns_rejected():
    df = pd.DataFrame({"image_id": ["a.jpg"], "label": [1]})
    with pytest.raises(ValueError):
        S._check_columns(df)


def test_unknown_site_rejected():
    df = _fake_meta()
    with pytest.raises(ValueError):
        S.leave_one_site_out(df, "SITE_DOES_NOT_EXIST")


def test_split_hash_is_stable_and_discriminating():
    df = _fake_meta()
    a = S.patient_level_split(df, seed=1)
    b = S.patient_level_split(df, seed=1)
    c = S.patient_level_split(df, seed=2)
    assert a.hash() == b.hash()
    assert a.hash() != c.hash()


def test_confounder_flag_fires_on_one_sided_site():
    df = _fake_meta()
    df.loc[df["site"] == "SITE_1", "label"] = 1        # make one site all-anemic
    balance = S.per_site_class_balance(df)
    assert bool(balance.loc["SITE_1", "confounder_flag"])
    assert not bool(balance.loc["SITE_0", "confounder_flag"])


# ---- metrics ---------------------------------------------------------------

def test_perfect_scores_give_auroc_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    rep = M.classification_report(y, p, spec_target=0.5)
    assert rep.auroc == pytest.approx(1.0)
    assert 0.0 <= rep.ece <= 1.0


def test_perfect_auroc_is_flagged_not_celebrated():
    """CLAUDE.md rule 4: AUROC ~ 1.00 must come back marked as a red flag."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    rep = M.classification_report(y, p, spec_target=0.5)
    assert rep.is_red_flag
    assert "leakage" in rep.note.lower()
    assert "RED FLAG" in rep.summary_line()


def test_report_never_exposes_accuracy():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    rep = M.classification_report(y, p, spec_target=0.5)
    assert "accuracy" not in rep.as_dict()
    assert "acc" not in rep.summary_line().lower()


def test_single_class_set_is_not_evaluable_rather_than_crashing():
    """Leave-one-site-out will hand us all-negative sites; that must produce a
    NaN plus an explanation, not an exception and not a fabricated number."""
    y = np.zeros(20, dtype=int)
    p = np.linspace(0.1, 0.9, 20)
    rep = M.classification_report(y, p)
    assert np.isnan(rep.auroc) and np.isnan(rep.auprc)
    assert "NOT EVALUABLE" in rep.note
    assert not rep.is_red_flag


def test_sensitivity_at_specificity():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    sens, thr = M.sensitivity_at_specificity(y, p, spec_target=0.9)
    assert 0.0 <= sens <= 1.0


def test_sensitivity_respects_the_specificity_floor():
    """The returned operating point must actually achieve the requested
    specificity -- that promise is the whole reason to report this metric."""
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 100)
    p = np.clip(np.where(y == 1, rng.normal(0.62, 0.16, 200),
                         rng.normal(0.38, 0.16, 200)), 0, 1)
    sens, thr = M.sensitivity_at_specificity(y, p, spec_target=0.90)
    achieved_spec = float((p[y == 0] < thr).mean())
    assert achieved_spec >= 0.90 - 1e-9
    assert float((p[y == 1] >= thr).mean()) == pytest.approx(sens, abs=1e-9)


def test_ece_is_zero_for_a_calibrated_model_and_large_for_a_liar():
    y = np.array([0] * 50 + [1] * 50)
    honest = np.array([0.05] * 50 + [0.95] * 50)
    liar = np.array([0.95] * 50 + [0.05] * 50)
    assert M.expected_calibration_error(y, honest) < 0.06
    assert M.expected_calibration_error(y, liar) > 0.85


def test_reliability_curve_drops_empty_bins():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.05, 0.95, 0.05, 0.95])
    xs, ys, ns = M.reliability_curve(y, p, n_bins=10)
    assert len(xs) == len(ys) == len(ns) == 2
    assert ns.sum() == len(y)


def test_generalisation_gap_reports_gap_and_skips_unevaluable_sites():
    good = M.classification_report(np.array([0, 0, 1, 1]),
                                   np.array([0.1, 0.2, 0.8, 0.9]),
                                   spec_target=0.5)
    weak = M.classification_report(np.array([0, 1, 0, 1]),
                                    np.array([0.4, 0.5, 0.6, 0.55]),
                                    spec_target=0.5)
    dead = M.classification_report(np.zeros(5, dtype=int), np.linspace(0, 1, 5))
    gap = M.generalisation_gap(good, [weak, dead], min_site_n=0)
    assert gap["n_sites_evaluated"] == 1 and gap["n_sites_skipped"] == 1
    assert gap["auroc_gap"] == pytest.approx(good.auroc - weak.auroc)


def test_generalisation_gap_excludes_tiny_sites_from_the_average():
    """A fold of 8 images cannot measure generalisation, and averaging it as an
    equal of a 134-image fold flatters the cross-site number."""
    strong = M.classification_report(np.tile([0, 1], 50),
                                     np.tile([0.2, 0.8], 50), spec_target=0.5)
    weak_big = M.classification_report(np.tile([0, 1], 50),
                                       np.r_[np.linspace(0.3, 0.7, 100)],
                                       spec_target=0.5)
    tiny_perfect = M.classification_report(np.array([0, 1, 1, 1]),
                                           np.array([0.1, 0.8, 0.9, 0.95]),
                                           spec_target=0.5)
    gap = M.generalisation_gap(strong, [weak_big, tiny_perfect], min_site_n=30)
    assert gap["n_sites_evaluated"] == 1
    assert gap["n_sites_too_small"] == 1
    # the tiny fold's perfect score must not enter the mean
    assert gap["site_out_auroc_mean"] == pytest.approx(weak_big.auroc)


def test_generalisation_gap_refuses_to_flatter_a_chance_level_model():
    """A near-zero gap under a chance-level in-distribution model means "bad
    everywhere", and must not be reportable as "generalises well"."""
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 60)
    chance = M.classification_report(y, rng.random(120), spec_target=0.5)
    other = M.classification_report(y, rng.random(120), spec_target=0.5)
    gap = M.generalisation_gap(chance, [other], min_site_n=0)
    assert "NOT INTERPRETABLE" in gap["note"]


def test_generalisation_gap_with_no_evaluable_site_says_so():
    good = M.classification_report(np.array([0, 0, 1, 1]),
                                   np.array([0.1, 0.2, 0.8, 0.9]),
                                   spec_target=0.5)
    dead = M.classification_report(np.ones(5, dtype=int), np.linspace(0, 1, 5))
    gap = M.generalisation_gap(good, [dead], min_site_n=0)
    assert np.isnan(gap["auroc_gap"])
    assert "unsupported" in gap["note"]


def test_regression_report_bland_altman():
    y = np.array([8.0, 9.5, 11.0, 12.5, 14.0])
    pred = y + 0.5                                  # a constant overestimate
    rep = M.regression_report(y, pred)
    assert rep.mae == pytest.approx(0.5)
    assert rep.bias == pytest.approx(0.5)
    assert rep.loa_low <= rep.bias <= rep.loa_high


# ---- colour features -------------------------------------------------------

def test_color_features_shape():
    img = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype("uint8")
    feats = color_features(img)
    vec = features_to_vector(feats)
    assert vec.shape == (len(FEATURE_ORDER),)


def test_color_features_detect_pallor_direction():
    """A paler patch must score lower on redness. If this ever inverts, the
    colour baseline is measuring something other than pallor."""
    red = np.zeros((16, 16, 3), dtype="uint8")
    red[..., 0], red[..., 1], red[..., 2] = 180, 90, 90
    pale = np.zeros((16, 16, 3), dtype="uint8")
    pale[..., 0], pale[..., 1], pale[..., 2] = 150, 110, 110
    assert color_features(pale)["redness"] < color_features(red)["redness"]
    assert color_features(pale)["sat_mean"] < color_features(red)["sat_mean"]


def test_roi_mask_drops_a_black_background():
    """CP-AnemiC images are conjunctiva strips pasted on black, and the
    background is most of the frame. Without this mask the colour features
    would encode the crop outline instead of the pallor."""
    img = np.zeros((20, 20, 3), dtype="uint8")
    img[5:15, :, :] = [180, 95, 95]                  # the "conjunctiva" band
    mask = roi_mask(img)
    assert mask.sum() == 10 * 20
    assert not mask[0, 0] and mask[10, 10]
    # r_mean over the ROI is the true strip colour, not diluted by the black
    assert color_features(img, mask)["r_mean"] == pytest.approx(180.0)
    assert color_features(img)["r_mean"] == pytest.approx(90.0)   # halved


def test_roi_mask_is_a_noop_on_an_unsegmented_photo():
    """Applying it unconditionally must be safe: a normal photo has almost no
    pure-black pixels, so nearly everything survives."""
    img = (np.random.default_rng(0).integers(40, 255, (32, 32, 3))).astype("uint8")
    assert roi_mask(img).mean() > 0.99


def test_color_features_respect_the_roi_mask():
    img = np.zeros((10, 10, 3), dtype="uint8")
    img[:5, :, 0] = 200                              # bright red top half only
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :] = True
    assert color_features(img, mask)["r_mean"] == pytest.approx(200.0)
    with pytest.raises(ValueError):
        color_features(img, np.zeros((10, 10), dtype=bool))


# ---- baselines -------------------------------------------------------------

def test_majority_baseline_is_uninformative_by_construction():
    """The floor: a constant predictor scores AUROC 0.5 and AUPRC = prevalence.
    Anything that cannot beat this has learned nothing."""
    y_tr = np.array([0] * 70 + [1] * 30)
    y_te = np.array([0] * 14 + [1] * 6)
    scores = B.majority_class_scores(y_tr, len(y_te))
    assert scores.min() == scores.max() == pytest.approx(0.3)
    rep = M.classification_report(y_te, scores)
    assert rep.auroc == pytest.approx(0.5)
    assert rep.auprc == pytest.approx(y_te.mean(), abs=0.02)


def test_colour_logistic_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 60)                          # interleaved: both classes
    X = rng.normal(0, 1, size=(120, len(FEATURE_ORDER)))
    X[y == 1, 0] += 3.0                              # planted separable feature
    scores, clf = B.fit_colour_logistic(X[:100], y[:100], X[100:], seed=0)
    assert scores.shape == (20,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert M.classification_report(y[100:], scores, spec_target=0.5).auroc > 0.9


def test_colour_logistic_refuses_a_single_class_training_fold():
    """Rather than fit an undefined discriminator, fall back to a constant."""
    X = np.random.default_rng(1).normal(0, 1, size=(30, len(FEATURE_ORDER)))
    y = np.ones(20, dtype=int)
    scores, _ = B.fit_colour_logistic(X[:20], y, X[20:])
    assert scores.min() == scores.max() == pytest.approx(1.0)


def test_site_prior_probe_exposes_a_site_confounder():
    """The probe uses no image data at all. If it scores well on a patient-level
    split, site identity alone predicts anemia and the dataset is confounded."""
    df = _fake_meta(n_patients=80, sites=4)
    df.loc[df["site"].isin(["SITE_0", "SITE_1"]), "label"] = 1
    df.loc[df["site"].isin(["SITE_2", "SITE_3"]), "label"] = 0
    s = S.patient_level_split(df)
    scores = B.site_prior_scores(df, s.train_idx, s.test_idx)
    y_te = df.loc[s.test_idx, "label"].to_numpy()
    assert M.classification_report(y_te, scores, spec_target=0.5).auroc > 0.95


def test_site_prior_probe_collapses_on_an_unseen_site():
    """Held-out sites have no training prevalence, so the probe must degrade to
    a constant -- site identity does not transfer to a new facility."""
    df = _fake_meta(n_patients=80, sites=4)
    s = S.leave_one_site_out(df, "SITE_0")
    scores = B.site_prior_scores(df, s.train_idx, s.test_idx)
    assert scores.min() == scores.max()


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", "-q", __file__])
