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

**Status: four experiments run on real CP-AnemiC data.** Baselines, confounder
audit, 35 ConvNeXt-Tiny runs across all 710 images and ten Ghanaian hospitals, a
positive control, and a hemoglobin-regression variant. Headline: on the
classification task the CNN does not beat an eleven-feature colour model in any
configuration; on the regression task it beats it by a small but paired-stable
margin, and neither is clinically usable. The shortcut explanation this project
first offered for the classification failure was tested and **refuted by its own
control**. See [Results](#results). The images themselves are never committed.

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
biology everywhere, so a model that learned it would not swing like that.

Grad-CAM (`reports/runs/*/gradcam.png`) showed the heat sitting on the black
background rather than on the conjunctiva, which suggested the model was reading
the hand-traced segmentation outline — a per-hospital fingerprint. **That
hypothesis was tested directly and it is wrong**; see the silhouette control
below. The variance is real, but the shortcut explanation for it did not
survive contact with evidence.

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

### Experiment 3: the silhouette control, and a retracted hypothesis

A diagnostic isolated the two interventions and found the collapse comes from
`randomise_background`, not from the augmentation: with the aggressive
augmentation switched off, predictions still span 0.0065.

Understanding why exposed a design error. **The background is uniform black in
all 710 images** — every background pixel measures 0–20 — so its colour cannot
distinguish one hospital from another. Only the traced outline varies.
Randomising the background colour therefore never touched the cue it was aimed
at, while replacing 72% of every input with noise that changes each epoch. The
intervention attacked the wrong variable. That is a reasoning error, not an
implementation one.

So the question was inverted: instead of subtracting the cue and watching the
model get worse, measure the cue on its own. `silhouette_only` trains the same
ConvNeXt-Tiny on **binary masks** — white where tissue is, black elsewhere, no
colour, no texture, no pallor. Whatever a hand-traced outline can predict, this
model gets to predict it.

| patient-level | AUROC | AUPRC | sens @ 90% spec |
|---|---|---|---|
| full image | 0.627 | 0.691 | 0.062 |
| background randomised + strong aug | 0.618 | 0.726 | 0.266 |
| background randomised only | 0.603 | 0.713 | 0.141 |
| **silhouette only (positive control)** | **0.530** | 0.645 | 0.125 |
| colour logistic | **0.668** | 0.729 | 0.219 |

**The silhouette alone is worthless: AUROC 0.530, bootstrap 95% CI
[0.404, 0.640], which contains chance.** The class means differ by 0.0014.

That refutes the shortcut hypothesis this project argued for two experiments
running. The hand-traced outline does not predict anemia, so "the CNN learned
the segmentation artefact" cannot explain either its weak performance or its
per-hospital variance. The Grad-CAM reading was over-interpretation — saliency
on a barely-better-than-chance model is not reliable evidence of what that model
uses.

The variance still needs an explanation, and the mundane one is now the leading
candidate: 28M parameters fitted to ~500 images are unstable, and which hospital
you hold out changes the draw.

One thing the control does support: a model handed genuinely uninformative input
collapses toward the base rate, exactly as the background-randomised runs did.
That is the correct behaviour, and it retroactively explains those collapses as
"no learnable signal survived the intervention" rather than an optimiser bug.

**Two hospitals contribute 8 and 15 images** and are reported but excluded from
cross-site averages. The colour baseline scored AUROC 1.000 on the 15-image one —
ranking two negatives — and the red-flag rule caught it automatically.

### Experiment 4: predicting hemoglobin instead of a label

Everything above asks "anemic yes or no". Regressing hemoglobin in g/dL is the
more general target, and it is what makes an external dataset usable at all: the
WHO cutoff is applied *after* the model, so one set of predictions serves
children at 11 g/dL and adults at 12 (women) / 13 (men). A binary model is
married to the threshold it was trained on. The binary label was also discarding
most of the signal — "anemic" in CP-AnemiC spans Hb 3.1 to 10.9, so a child
needing transfusion and one a tenth below the cutoff carried the same target.

The floor is no longer the majority class but **predicting the training mean**,
which on this data is 1.73–1.80 g/dL depending on the split.

| | patient-level MAE | cross-site MAE (8 hospitals) | vs. trivial |
|---|---|---|---|
| ConvNeXt-Tiny | 1.74 | **1.68** | 0.93 |
| colour ridge | **1.69** | 1.78 | 0.99 |
| predict the mean | 1.73 | 1.80 | 1.00 |

**In this framing the CNN beats the colour baseline cross-site — the reverse of
the classification result.** Paired across the same eight folds the difference is
+0.102 g/dL in its favour, winning 5 of 8, 95% CI [+0.010, +0.193], which
excludes zero but barely. The ridge penalty was swept over five orders of
magnitude (cross-site MAE moves only 1.783 → 1.799), so the baseline is not
crippled and the gap is real.

Two things stop that being a win:

**The CNN does better cross-site (1.68) than patient-level (1.74)**, which is
backwards — the harder split should not beat the easier one. The mundane
explanation is that each leave-one-site-out fold trains on ~600 images against
the patient-level split's 498. A 20% data increase moving the result this much
says the model is data-starved, which supports "more data, not a bigger model"
rather than "the model works".

**It is still clinically useless.** MAE 1.68 g/dL with Bland-Altman limits of
agreement near ±4 g/dL, against a scale running 3 to 17. Being 6% better than
colour and 7% better than guessing the mean does not make a measurement.

The regression view also exposes how weak the signal is in a way AUROC hid.
AUROC 0.668 reads as "weak but real"; ±1.7 g/dL reads as what it is. The two do
not conflict — AUROC measures ranking, MAE measures magnitude, and colour can
rank slightly better than chance without estimating the value.

*A hypothesis tested and dropped:* the cross-site improvement was first
attributed to an easier trivial floor (a held-out site far from the global mean
inflates the denominator). The correlation between a site's deviation from the
global mean and its MAE ratio is **+0.35** — the opposite direction to what that
explanation predicts. At n=8 it settles nothing, but it does not support it.

## Case study

### Problem

Anemia affects over a billion people and hits hardest where diagnosis is
hardest: confirming it needs a blood draw, a lab, and a trained technician.
Clinicians already read the palpebral conjunctiva by eye — it pales as
hemoglobin falls — so a smartphone photo of it is an obvious candidate for
triage in a clinic that has no analyser. The literature is full of papers
reporting high accuracy on that task. The question this project asked was not
"can we match them" but "would the number survive a hospital the model has never
seen", because a screening tool that only works where it was trained is not a
screening tool.

### Decisions

**Baselines before the network, enforced in code.** Pallor is a colour change,
so eleven colour statistics plus logistic regression is a serious competitor,
not a straw man. `02_train.py` exits non-zero unless the baseline file exists,
and prints the colour number beside every CNN number it produces. There is no
code path that yields a standalone deep-learning result.

**Two splits, always reported together.** Patient-level (optimistic) and
leave-one-site-out over all ten hospitals (honest). `assert_no_leakage` runs
inside both split functions, so a leaky split cannot be built by accident.

**A confounder probe with no pixels in it.** `site_prior` predicts a child's
label from the collection site alone. If site identity is predictive, every
in-distribution number is partly measuring "which facility is this".

**Metrics a clinician would ask for.** AUROC, AUPRC, sensitivity at 90%
specificity, and calibration. `classification_report` does not return accuracy
at all — on a 60/40 split it would flatter every model here. AUROC ≥ 0.99 comes
back tagged as a suspected leak rather than a win.

### Result

Twenty-four training runs across four configurations. The finding is negative
and it is stable:

| patient-level | cross-site (8 hospitals) | |
|---|---|---|
| colour logistic, 11 features | **0.668** | **0.593** |
| ConvNeXt-Tiny, 27.8M params | 0.627 | 0.603 |
| `site_prior`, zero pixels | 0.660 | 0.500 |
| majority class | 0.500 | — |

**A pretrained ConvNeXt-Tiny does not beat eleven colour statistics** — not on
the optimistic split, not across hospitals, and not in any of the four
preprocessing variants tried (0.627 / 0.618 / 0.603). At 90% specificity it
catches 6% of anemic children against colour's 22%. Neither is deployable.

**Knowing the hospital is nearly as predictive as seeing the photo** (0.660 vs
0.668), and that signal vanishes on an unseen site while the colour signal
survives at 0.593. There is real pallor information here; it is just weak.

Three decisions changed what got reported, and all three were caught by
measurement rather than intuition:

- The images are pre-segmented strips on black, with the background covering
  52–92% of the frame and varying 40 points between images. Unmasked, the colour
  features encoded the crop outline instead of the pallor.
- Two hospitals contribute 8 and 15 images and scored AUROC 0.857 and 1.000.
  Averaging them as equals of a 134-image fold reported the generalisation gap
  as +0.008; excluding them gives **+0.075**. That was the difference between a
  flattering number and a true one.
- **The project's own headline explanation turned out to be wrong.** Grad-CAM
  showed attention on the black background, and two experiments were built on
  the theory that the CNN was reading the hand-traced segmentation outline. A
  positive control — the same network trained on binary silhouettes, no colour,
  no texture — scored AUROC 0.530 with a bootstrap CI of [0.404, 0.640]. The
  outline predicts nothing. The saliency reading was over-interpretation, and
  the remaining explanation for the CNN's 0.399–0.871 swing across hospitals is
  the mundane one: 28M parameters on ~500 images are unstable.

### Limitations

This is a screening signal, not a diagnosis; lab hemoglobin remains the
reference standard, and nothing here is a medical device.

The population is the dataset's own — children aged 6–59 months in ten Ghanaian
hospitals — and no claim extends past it. 710 images is small for deep learning,
which is itself part of the finding rather than an excuse: at this scale a
27.8M-parameter network has no advantage over a linear model on eleven features.

Every result is single-seed. With this much data, seed-to-seed variation is
plausibly ±0.05 AUROC, so the honest statement is "the CNN fails to beat colour",
not "colour beats the CNN". Multiple seeds per configuration would settle it.

The cross-site average rests on eight hospitals, two of which are excluded as
too small to measure anything. Claiming this generalises would need a second
dataset from a different country, different phones, and a different segmentation
protocol — the last of which we now know matters less than suspected, but only
because it was tested.

What would move the needle is more data, not a bigger model. That conclusion is
the opposite of where this project started, and it is the one the measurements
support.

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
tests/          test_smoke.py  (47 tests: splits, leakage, metrics, baselines,
                ROI masking, shortcut controls, Hb regression — no data, no GPU)
config/         default.yaml · cpanemic.yaml · cpanemic_hb.yaml
                cpanemic_matched.yaml · cpanemic_bgonly.yaml
                cpanemic_silhouette.yaml · synthetic.yaml
reports/        baselines/ · runs/ · summary.md   (git-ignored)
.claude/agents/ six specialist subagents
CLAUDE.md       the rigor doctrine
```

## License / data

Code: MIT (see `LICENSE`). The dataset is **not** covered by it, is not included
and is never committed — these are clinical photographs of children aged 6–59
months. Get them from the original source and respect its terms:

> Appiahene et al., "CP-AnemiC: A conjunctival pallor dataset and benchmark for
> anemia detection in children", *Medicine in Novel Technology and Devices* 18
> (2023) 100244.

Nothing here is a medical device or a diagnosis.
