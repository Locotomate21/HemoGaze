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

**Status: pipeline complete and verified end to end; no real-data results yet.**
The CP-AnemiC images are not in this repo and have not been run through it. Every
number you can currently produce comes from a deliberately fabricated stand-in
dataset and is stamped `SYNTHETIC`. See [Results](#results).

## Why this repo looks the way it does

The interesting question in medical imaging is rarely "what accuracy did you
get" — it's "did the model learn the biology, or the camera?" So the
architecture of the project is built around that question:

- **Two splits, always reported together:** a patient-level split (optimistic)
  and a **leave-one-site-out** sweep over every site (honest). The gap between
  them is the headline result.
- **A real baseline first:** majority-class, then an eleven-feature colour
  logistic regression. Pallor is a colour change, so if a heavy CNN cannot
  clearly beat colour, that is a finding — not something to hide.
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
pip install -r requirements.txt        # torch/timm only needed from step 3

# 1. verify the honest core with no data and no GPU
pytest -q

# 2. OPTIONAL: fabricate a stand-in dataset to exercise the whole chain on CPU
python scripts/make_synthetic_data.py --out data/synthetic
#    then use config/synthetic.yaml in place of config/default.yaml below

# 3. get the real data  ->  see data/README.md  (CP-AnemiC + a metadata.csv)

# 4. EDA, confounder audit, and baselines (run and read BEFORE any neural net)
python scripts/00_eda_baseline.py --config config/default.yaml

# 5. train
python scripts/02_train.py --config config/default.yaml                # patient-level
python scripts/02_train.py --config config/default.yaml --all-sites    # cross-site sweep

# 6. figures per run, then the comparison table
python scripts/03_evaluate.py --run reports/runs/baseline_patient_level
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

There are none yet on real data, and this section will not be filled in with
anything else.

The pipeline has been verified end to end on the synthetic stand-in
(720 images, 360 patients, 10 sites, two fabricated sites near-single-class).
That exercise confirms the machinery behaves as designed:

- the confounder audit flagged both one-sided sites;
- the `site_prior` probe scored AUROC 0.69 on the patient-level split and
  collapsed to 0.50 on every unseen site — the fabricated site confounder,
  correctly detected and correctly shown not to transfer;
- the colour baseline showed a positive patient-level → cross-site gap, with
  AUPRC degrading further than AUROC on the imbalanced folds;
- a deliberately under-trained CNN came out **below** the colour baseline, and
  the script said so in those words.

Those are statements about the code. They are not findings about anemia, and
the synthetic generator exists precisely so that nobody has to pretend
otherwise while the real data is unavailable.

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
scripts/        00_eda_baseline → 02_train → 03_evaluate
                make_synthetic_data.py  (plumbing, not pipeline)
tests/          test_smoke.py  (30 tests: splits, leakage, metrics, baselines —
                no data, no GPU)
config/         default.yaml · synthetic.yaml
reports/        baselines/ · runs/ · summary.md   (git-ignored)
.claude/agents/ six specialist subagents
CLAUDE.md       the rigor doctrine
```

## License / data

Code: add your license. Data: not included and never committed — patient
images are sensitive (see `.gitignore` and `data/README.md`).
