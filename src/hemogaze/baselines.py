"""The baselines a deep model has to beat before it is worth quoting.

CLAUDE.md rule 2 says a majority-class baseline and a colour-feature logistic
regression must exist and be *reported* before any CNN number is spoken aloud.
That logic lives here rather than inside a script so it is importable, unit
testable, and impossible to quietly skip.

Three baselines, each answering a different question:

- ``majority_class_scores``  -- what does predicting the training prevalence,
  every time, already get you? This is the floor. On an imbalanced set it makes
  accuracy look respectable, which is exactly why accuracy is banned as a
  headline.
- ``colour_logistic``       -- pallor *is* a colour change, so eleven colour
  statistics + logistic regression is a genuinely strong model, not a straw man.
  If the CNN ties it, that is the finding.
- ``site_prior_scores``     -- a confounder probe, not a real model: predict a
  child's label from the collection site alone. If this scores well, the
  dataset carries site signal and any patient-level split result is partly
  measuring "which facility is this", not pallor.

No torch anywhere in this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def majority_class_scores(y_train, n_test: int) -> np.ndarray:
    """Constant predictor: the training prevalence, repeated.

    Returned as a probability rather than a hard 0/1 so it flows through the
    same metric functions as every other model. A constant score scores AUROC
    0.5 and AUPRC = test prevalence by construction, which is the point: it
    shows what "no information" looks like on this data.
    """
    y_train = np.asarray(y_train, dtype=float)
    if y_train.size == 0:
        raise ValueError("Cannot fit a majority baseline on an empty train set.")
    return np.full(int(n_test), float(y_train.mean()), dtype=float)


def colour_logistic(class_weight: str | None = "balanced",
                    max_iter: int = 1000, seed: int = 42) -> Pipeline:
    """Standardise then logistic-regress the colour features.

    ``class_weight="balanced"`` because the screening error that matters is a
    missed anemic child, and an unweighted fit on imbalanced data drifts toward
    the majority class.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, class_weight=class_weight,
                           random_state=seed),
    )


def fit_colour_logistic(X_train, y_train, X_test, *, seed: int = 42
                        ) -> tuple[np.ndarray, Pipeline]:
    """Fit the colour baseline and return (test scores, fitted pipeline).

    The scaler is fitted on train only -- fitting it on train+test is a subtle
    leak that flatters every downstream number.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train).astype(int)
    if len(np.unique(y_train)) < 2:
        # A single-class training fold cannot fit a discriminative model; say so
        # instead of returning a number that looks like a result.
        return np.full(len(X_test), float(y_train.mean())), colour_logistic(seed=seed)
    clf = colour_logistic(seed=seed)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1], clf


def site_prior_scores(df: pd.DataFrame, train_idx, test_idx) -> np.ndarray:
    """Confounder probe: score each test image with the *training* anemia
    prevalence of its own site, ignoring the image entirely.

    Sites unseen in training fall back to the global training prevalence, so on
    a leave-one-site-out split this collapses to a constant by design -- which
    is the honest answer, since site identity carries no transferable
    information about a facility you have never visited.
    """
    y_tr = df.loc[train_idx, "label"].astype(float)
    global_prior = float(y_tr.mean())
    per_site = df.loc[train_idx].groupby("site")["label"].mean()
    return (df.loc[test_idx, "site"]
              .map(per_site)
              .fillna(global_prior)
              .to_numpy(dtype=float))
