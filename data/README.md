# Data

This project uses **CP-AnemiC**, a public conjunctival-pallor dataset for
anemia detection in children: ~710 conjunctiva images collected across ten
health facilities in Ghana, each labelled with hemoglobin (g/dL), age, sex,
and **collection site**, with the WHO cutoff (Hb < 11 g/dL) defining anemia.

## How to get it

1. Find the dataset via its publication (search: **"CP-AnemiC conjunctival
   pallor dataset"**) and follow the *Data availability* link in the paper.
2. Download it into `data/cp-anemic/`.
3. Produce a `metadata.csv` with **at least** these columns:

   | column       | meaning                                   |
   |--------------|-------------------------------------------|
   | `image_id`   | filename relative to `data_dir`           |
   | `patient_id` | one id per child (both eyes share it)     |
   | `site`       | collection facility (the confounder axis) |
   | `label`      | 1 = anemic (Hb < 11), 0 = not             |
   | `hemoglobin` | g/dL (optional, enables the Hb-regression variant) |
   | `age_months` | optional                                  |
   | `sex`        | optional                                  |

If the original files use different column names, write a tiny adapter script
that renames them into the schema above.

## Testing the pipeline without the real data

`scripts/make_synthetic_data.py` fabricates a stand-in dataset in this exact
schema — 720 images, 360 patients, ten sites, with a weak pallor signal and a
strong site confounder built in on purpose:

```bash
python scripts/make_synthetic_data.py --out data/synthetic
python scripts/00_eda_baseline.py --config config/synthetic.yaml
```

It writes a `SYNTHETIC.json` marker that every script checks, so its outputs are
stamped and cannot later be mistaken for results. Nothing produced from it says
anything about anemia — it only proves the code does what it claims.

## Ethics / honesty notes

- Patient images are sensitive. **Nothing under `data/` is committed** (see
  `.gitignore`). Do not upload raw images anywhere public.
- This is a **screening / triage signal, not a diagnosis**. Hemoglobin from a
  blood test remains the reference standard.
- The dataset is a specific population (young children in Ghana). Any claim
  beyond that population is unsupported until you have external data.
