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

**Status: full experiment run on real CP-AnemiC data.** Baselines, confounder
audit, and eleven ConvNeXt-Tiny runs across all 710 images and ten Ghanaian
hospitals. Headline: the CNN does not beat an eleven-feature colour model, and
its per-hospital variance is 2.5× the baseline's. See [Results](#results). The
images themselves are never committed.

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

Real CP-AnemiC: 710 images, 710 rows, ten Ghanaian hospitals, 59.7% anemic.
Eleven runs: one patient-level, one per held-out hospital. Seed 42,
`split_hash` recorded with every number.

| | patient-level | cross-site (8 hospitals) | gap |
|---|---|---|---|
| **ConvNeXt-Tiny**, 27.8M params | 0.627 | 0.603 | +0.024 |
| **colour logistic**, 11 features | **0.668** | 0.593 | +0.075 |
| `site_prior` probe (zero pixels) | 0.660 | 0.500 | — |
| majority class (the floor) | 0.500 | — | — |

**A pretrained ConvNeXt-Tiny does not beat eleven colour statistics.** It loses
on the patient-level split (−0.041 AUROC) and ties across hospitals. At 90%
specificity it catches 6% of anemic children against the colour model's 22%.
Neither is deployable; the expensive one is worse.

**Knowing which hospital took the photo is almost as predictive as the photo.**
The `site_prior` probe uses no pixels and reaches 0.660, against colour's 0.668.
It collapses to chance on an unseen hospital while the colour signal survives at
0.593 — so there is real pallor signal, it is just weak.

**The dispersion matters more than the mean.** Across the eight evaluable
hospitals:

| | mean | min | max | std | range |
|---|---|---|---|---|---|
| ConvNeXt-Tiny | 0.603 | 0.399 | 0.871 | 0.163 | 0.472 |
| colour logistic | 0.593 | 0.515 | 0.701 | 0.068 | 0.186 |

Same mean, **2.5× the spread**. The CNN scores 0.871 at one hospital and 0.399 —
below chance — at another, falling under 0.5 in two of eight. Pallor is the same
biology everywhere, so a model that learned it would perform similarly
everywhere. That variance is the signature of a shortcut, and Grad-CAM
(`reports/runs/*/gradcam.png`) shows the heat sitting on the black background
and the crop silhouette rather than on the conjunctiva.

**Known asymmetry in experiment 1.** The colour baseline gets `roi_mask`, so it
cannot see the segmentation outline; the CNN was fed the raw frame and can. The
silhouette was hand-traced per hospital, so the confounder blocked for the
baseline was left available to the CNN. Experiment 2 below was run to close that
gap.

### Experiment 2: the matched rerun (partially failed, reported anyway)

Same seed, splits, backbone and resolution, with the segmentation background
repainted a random colour on every training view (`fill_background`) plus
stronger geometric augmentation. Two other removal mechanisms were tried first
and rejected on measurement, not on taste: cropping to the ROI bounding box
changes nothing here (a conjunctiva is a crescent whose bounding box is the whole
frame — background stayed at a median 72% either way), and tissue-only patch
sampling is infeasible (a window at 50% of the shorter side reaches ≥90% tissue
in only 18% of images).

| | mean | range | std |
|---|---|---|---|
| experiment 1 (raw frame) | 0.603 | 0.399–0.871 | 0.163 |
| experiment 2, all folds | 0.599 | 0.510–0.725 | 0.070 |
| experiment 2, non-collapsed folds only | 0.589 | 0.510–0.725 | 0.087 |
| colour logistic | 0.593 | 0.515–0.701 | 0.068 |

**The intervention broke training in three of eight evaluable folds.** In those
runs every prediction lands inside a band of 0.004–0.025 centred on the training
prevalence: the model collapsed to predicting the base rate. Those folds still
report an AUROC, built on differences of ~0.001 between scores, and it should not
be read as a measurement. The patient-level run collapsed too, which also means
its apparently excellent ECE of 0.028 is an artefact — a constant predictor at
the base rate is calibrated by construction, not by merit.

So experiment 2 does **not** cleanly answer the question it was built for. What
survives it:

- **The headline is unchanged and now triply confirmed.** 0.603, 0.599, 0.589 —
  every way of slicing it lands on the colour baseline's 0.593. The CNN ties an
  eleven-feature logistic regression whether or not the shortcut is available.
- **The dispersion claim is weaker than the all-folds table suggests.** Comparing
  only the five folds that trained properly, the spread falls from std 0.132 to
  0.087 — a real reduction, but half the apparent effect, and n=5 makes it soft.

A diagnostic isolating the two interventions (background randomisation without
the aggressive augmentation) is what would tell us whether the collapse is
fixable and the experiment worth rerunning.

**Two hospitals contribute 8 and 15 images** and are reported but excluded from
cross-site averages. The colour baseline scored AUROC 1.000 on the 15-image one —
ranking two negatives — and the red-flag rule caught it automatically.

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
