# HemoGaze — project instructions for Claude Code

Non-invasive **anemia screening from conjunctiva images**, built with the same
discipline as an honest ML forecasting pipeline: real baselines, correct
metrics, cross-site validation, and no overselling.

## What this project is (and is not)

- It IS a screening/triage signal from a smartphone photo of the palpebral
  conjunctiva.
- It is NOT a diagnosis. Lab hemoglobin is the reference standard.
- The dataset (CP-AnemiC) is young children in Ghana across ten sites. Do not
  let any claim exceed that population.

## Non-negotiable rules (enforce these in every change)

1. **Split by patient, then by site.** No patient (and no eye of that patient)
   may appear on two sides of a split. Always report BOTH a patient-level
   split and a leave-one-site-out split. `assert_no_leakage` must pass.
2. **Baseline before neural net.** A majority-class baseline and a
   colour-feature logistic regression must exist and be reported before any
   CNN result is quoted. If the CNN barely beats colour logistic regression,
   say so plainly.
3. **Metrics: never headline accuracy.** Report AUROC, AUPRC,
   sensitivity-at-fixed-specificity, and calibration (reliability diagram +
   ECE). For the Hb-regression variant use MAE and Bland-Altman.
4. **AUROC ≈ 1.00 is a red flag, not a trophy.** Treat any near-perfect score
   as suspected leakage or a site/camera confounder until proven otherwise.
5. **Confounder vigilance.** The generalisation gap between the patient-level
   and leave-one-site-out results is the headline finding. Audit per-site
   class balance; be suspicious of a site that is almost all one class.
6. **Reproducibility.** Seed everything; save the resolved config and the
   split hash next to every metrics file.
7. **No overselling in prose.** READMEs, commit messages, posts, and abstracts
   describe limitations and scope honestly. No "perfect", no "diagnoses", no
   dropping the population caveat.

## Layout

- `src/hemogaze/` — importable modules (`splits`, `metrics`, `features`,
  `baselines`, `dataset`, `model`, `config`). Everything except `dataset` and
  `model` has **no torch dependency** and is covered by `tests/test_smoke.py`;
  those two are the torch boundary and are never imported by `__init__`.
- `scripts/` — `00_eda_baseline.py` → `02_train.py` → `03_evaluate.py`, in
  order. `make_synthetic_data.py` is unnumbered because it is plumbing, not
  pipeline: it fabricates a stand-in dataset so the whole chain can be run
  without patient images.
- `config/default.yaml` — all hyperparameters. `config/synthetic.yaml` is the
  CPU smoke-run counterpart.
- `reports/baselines*/` — step-0 output; `02_train.py` **refuses to run**
  without it. `reports/runs*/` — per-run config, weights, metrics, figures.
- `.claude/agents/` — specialist subagents; see below.

## Conventions

- Python, functional and explicit; type hints; docstrings that say *why*.
- `pytest -q` must stay green. New data/split/metric logic needs a test.
- Run `python scripts/00_eda_baseline.py` and read its confounder audit before
  training anything.
- Any artifact produced from the synthetic stand-in is stamped `SYNTHETIC`
  (`Config.is_synthetic()`). Never quote a synthetic number as a result.
- `classification_report` returns NaN plus an explanatory `note` for
  single-class evaluation sets instead of raising — leave-one-site-out will
  hand you those, and "not evaluable" is itself a finding to report.

## Subagents (delegate deliberately)

- **data-integrity** — before any training / whenever splits or metadata change.
- **baseline-guardian** — before quoting any deep-model result.
- **metrics-honesty** — whenever metrics are computed, reported, or written up.
- **model-architect** — when designing/altering the architecture.
- **experiment-runner** — to launch reproducible training/eval runs.
- **case-study-writer** — for the README case study, LinkedIn post, or abstract.
