# HemoGaze

**Non-invasive anemia screening from conjunctiva images — with honest cross-site validation.**

Anemia affects over a billion people, and confirming it needs a blood draw and
a lab. The palpebral conjunctiva (inner lower eyelid) pales as hemoglobin
drops — a sign clinicians already read by eye. HemoGaze asks whether a
smartphone photo of it gives a useful *screening* signal, and holds itself to
the standard that matters: does it still work on a health facility it has never
seen?

> Screening signal, **not** a diagnosis. Reference standard is lab hemoglobin.
> Dataset is young children in Ghana across ten sites; claims stop there.

**Status: baselines run on real CP-AnemiC data; CNN not yet trained.**
The colour baseline and the confounder audit have been run on all 710 images
across ten Ghanaian hospitals. The deep model has not. See [Results](#results).
The images themselves are never committed.

## Why this repo looks the way it does

The interesting question in medical imaging is rarely "what accuracy did you
get" — it's "did the model learn the biology, or the camera?" So the
architecture of the project is built around that question:

- **Two splits, always reported together:** a patient-level split (optimistic)
  and a **leave-one-site-out** sweep over every site (honest). The gap between
  them is the headline result.
- **A real baseline first:** majority-class, then an eleven-feature colour
  logistic regression, taken over an ROI mask rather than the whole frame.
  Pallor is a colour change, so if a heavy CNN cannot clearly beat colour, that
  is a finding — not something to hide.
- **A confounder probe:** `site_prior` predicts a child's label from the
  collection site alone, using no pixels at all. If it scores well on the
  patient-level split, site identity itself predicts anemia and every
  in-distribution number is partly measuring "which facility is this".
- **Screening-appropriate metrics:** AUROC, AUPRC, sensitivity at a fixed
  specificity, and **calibration** (reliability diagram + ECE). Never headline
  accuracy — `classification_report` does not even return it.
- **A rule you will see enforced everywhere:** AUROC ≥ 0.99 comes back tagged
  as a leakage red flag, not a trophy.

Two of those rules are enforced by code rather than by good intentions:

| rule | how it is enforced |
|---|---|
| baselines before the neural net | `02_train.py` **exits** unless `reports/baselines/baseline_metrics.json` exists, and prints the colour baseline for the same split next to every CNN number |
| no patient or site leakage | `assert_no_leakage` runs inside both split functions, so a leaky split cannot be constructed by accident |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # torch/timm only needed from step 5

# 1. verify the honest core with no data and no GPU
pytest -q

# 2. OPTIONAL: fabricate a stand-in dataset to exercise the whole chain on CPU
python scripts/make_synthetic_data.py --out data/synthetic
#    then use config/synthetic.yaml in place of config/default.yaml below

# 3. get the real data (see data/README.md), then adapt it to the schema
python scripts/01_prepare_metadata.py --src "path/to/CP-AnemiC dataset"

# 4. EDA, confounder audit, and baselines (run and read BEFORE any neural net)
python scripts/00_eda_baseline.py --config config/cpanemic.yaml

# 5. train
python scripts/02_train.py --config config/cpanemic.yaml               # patient-level
python scripts/02_train.py --config config/cpanemic.yaml --all-sites   # cross-site sweep

# 6. figures per run, then the comparison table
python scripts/03_evaluate.py --run reports/runs/convnext128_patient_level
python scripts/03_evaluate.py --compare        # -> reports/summary.md
```

Step 4 alone already gives you the generalisation gap, for the colour baseline,
before a single GPU hour is spent. That is the cheapest honest result in the
project and it is available on day one.

### What each step produces

- **`00_eda_baseline.py`** — per-site class balance with confounder flags,
  sites that are single-class and therefore not evaluable at all, and all three
  baselines on the patient-level split plus every leave-one-site-out fold.
  Writes `reports/baselines/baseline_metrics.json` and the resolved config.
- **`02_train.py`** — ConvNeXt-Tiny (or EfficientNet-B0) with a head-only
  warmup, positive-class-weighted BCE, and model selection on **AUPRC**. Saves
  weights, predictions, the resolved config, the split hash, the trainable
  parameter count, and the CNN-minus-baseline delta.
- **`03_evaluate.py`** — reliability diagram with bin counts, score
  distribution by true class, Grad-CAM overlays on held-out images (plain
  autograd hooks, no extra dependency), and `reports/summary.md`.

## Results

Real CP-AnemiC: 710 images, ten hospitals, 59.7% anemic. Colour baseline only —
**no CNN has been trained yet**, so nothing below is a deep-learning result.

| | AUROC | AUPRC | sens @ 90% spec | ECE |
|---|---|---|---|---|
| majority class (the floor) | 0.500 | 0.604 | 0.000 | — |
| colour logistic, patient-level | **0.668** | 0.729 | 0.219 | 0.110 |
| colour logistic, unseen hospital (8 sites) | **0.593** | — | — | — |
| **generalisation gap** | **+0.075** | +0.042 | | |
| `site_prior` probe, patient-level | 0.660 | 0.735 | 0.250 | 0.036 |
| `site_prior` probe, unseen hospital | 0.500 | — | — | — |

Three things worth stating plainly:

**Knowing which hospital took the photo is almost as predictive as the photo.**
The `site_prior` probe uses zero pixels and reaches 0.660 against the colour
model's 0.668. On the optimistic split the image adds very little over the site
confounder. The colour signal does transfer (0.593 on unseen hospitals) while
site identity collapses to chance, so there is real pallor signal — it is just
weak.

**This is not yet a usable screening tool.** At 90% specificity the colour model
catches 22% of anemic children. Four in five are missed.

**Two hospitals contribute 8 and 15 images** and scored AUROC 0.857 and 1.000.
Those folds cannot measure anything, and averaging them as equals of a
134-image fold moved the reported gap from +0.075 to +0.008. They are reported
but excluded from cross-site averages.

The pipeline was verified end to end on a synthetic stand-in before the real
data arrived; that exercise is a statement about the code, not about anemia.

## Case study (fill in as real results land)

**Problem.** _One paragraph: the screening gap anemia detection has._

**Decisions.** _Patient- and site-level splits; colour baseline; screening
metrics + calibration. Why each._

**Result.** _Patient-level vs leave-one-site-out, side by side. Baseline vs
CNN. Lead with the generalisation gap. Cite the split_hash._

**Limitations.** _Screening not diagnosis; one population; small data; what
external data would be needed to claim more._

## Working with Claude Code

`CLAUDE.md` holds the non-negotiable rigor rules. Six specialist subagents live
in `.claude/agents/` and are meant to be delegated to:

| agent | when it runs |
|---|---|
| `data-integrity` | before training / whenever splits or metadata change |
| `baseline-guardian` | before quoting any deep-model result |
| `metrics-honesty` | whenever metrics are computed or written up |
| `model-architect` | when designing/altering the architecture |
| `experiment-runner` | to launch reproducible runs + the cross-site loop |
| `case-study-writer` | for the README case study, a post, or an abstract |

## Layout

```
src/hemogaze/   splits · metrics · features · baselines · dataset · model · config
                (only dataset + model import torch)
scripts/        01_prepare_metadata → 00_eda_baseline → 02_train → 03_evaluate
                make_synthetic_data.py  (plumbing, not pipeline)
tests/          test_smoke.py  (33 tests: splits, leakage, metrics, baselines,
                ROI masking — no data, no GPU)
config/         default.yaml · cpanemic.yaml · synthetic.yaml
reports/        baselines/ · runs/ · summary.md   (git-ignored)
.claude/agents/ six specialist subagents
CLAUDE.md       the rigor doctrine
```

## License / data

Code: add your license. Data: not included and never committed — patient
images are sensitive (see `.gitignore` and `data/README.md`).
