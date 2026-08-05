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

The repo is the `genIA_services` monorepo; this project lives in
`Python/hemogaze/` and **all commands run from that directory**.

- `processing/` — domain logic with **no torch dependency**: `features`
  (ROI masking, colour statistics), `splits`, `metrics`, `baselines`, `config`,
  plus the dataset adapters `prepare_cpanemic.py` and `prepare_external.py`.
- `training/` — the torch boundary and the pipelines: `dataset`, `model`,
  `eda_baseline` → `train` → `evaluate`, `external_validation`,
  `make_synthetic_data` (plumbing: a stand-in dataset so the chain runs with no
  patient images), and `configs/*.yaml`.
- `server/` — FastAPI service. Returns g/dL, never a label, and carries
  `valid_for_screening: false` plus the evidence in every response.
- `examples/` — CLI and web clients; they depend on nothing in `training/`.
- `reports/` and `Datasets/` — **DVC-managed, on DagsHub.** Only the ~150-byte
  `.dvc` pointers are in git. `reports/baselines*/` is step-0 output and
  `train.py` **refuses to run** without it.
- `.claude/agents/` — specialist subagents; see below.

Imports are flat (`from features import roi_mask`) because `WORKDIR` is the
project root in both Dockerfiles; the scripts add `processing/` and `training/`
to `sys.path` for local runs.

## Conventions

- Python, functional and explicit; type hints; docstrings that say *why*.
- `pytest -q` must stay green. New data/split/metric logic needs a test.
- Run `python training/eda_baseline.py` and read its confounder audit before
  training anything.
- Any artifact produced from the synthetic stand-in is stamped `SYNTHETIC`
  (`Config.is_synthetic()`). Never quote a synthetic number as a result.
- `classification_report` returns NaN plus an explanatory `note` for
  single-class evaluation sets instead of raising — leave-one-site-out will
  hand you those, and "not evaluable" is itself a finding to report.

## What CP-AnemiC actually turned out to be (verified, not assumed)

- **710 images, 710 rows, ten hospitals.** Labels are perfectly consistent with
  Hb < 11 g/dL; the `Anemic/` `Non-anemic/` folders agree on all 710.
- **The images are pre-segmented conjunctiva strips on black.** Background is
  52–92% of the frame and varies 40 points between images, so colour features
  **must** be taken over `roi_mask(img)`. Unmasked they encode the crop outline,
  not the pallor.
- **One image per child, verified against the publication.** There is no
  patient identifier column, so `patient_id = image_id`; the paper states 710
  individuals / 710 participants and heads its Table 2 "patient-level
  characteristics", and the metadata reproduces its 306/404 sex split and mean
  age to two decimals. The patient-level split carries no leakage on this axis.
- **Two hospitals contribute 8 and 15 images.** They are reported but excluded
  from cross-site averages (`MIN_SITE_N`); including them moved the gap from
  +0.075 to +0.008 and would have been the difference between an honest finding
  and a flattering one.

## Subagents (delegate deliberately)

- **data-integrity** — before any training / whenever splits or metadata change.
- **baseline-guardian** — before quoting any deep-model result.
- **metrics-honesty** — whenever metrics are computed, reported, or written up.
- **model-architect** — when designing/altering the architecture.
- **experiment-runner** — to launch reproducible training/eval runs.
- **case-study-writer** — for the README case study, LinkedIn post, or abstract.
