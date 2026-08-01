# Data

This project uses **CP-AnemiC**, a public conjunctival-pallor dataset for
anemia detection in children: ~710 conjunctiva images collected across ten
health facilities in Ghana, each labelled with hemoglobin (g/dL), age, sex,
and **collection site**, with the WHO cutoff (Hb < 11 g/dL) defining anemia.

## How to get it

1. Find the dataset via its publication (search: **"CP-AnemiC conjunctival
   pallor dataset"**) or on Kaggle.
2. Unzip it anywhere. The distribution looks like this:

   ```
   CP-AnemiC dataset/
       Anemia_Data_Collection_Sheet.xlsx    710 rows of clinical metadata
       Anemic/          Image_001.png ...   424 files
       Non-anemic/      Image_003.png ...   286 files
   ```

3. Run the adapter, which reconciles the spreadsheet against the folders and
   writes the schema below:

   ```bash
   python scripts/01_prepare_metadata.py --src "path/to/CP-AnemiC dataset"
   ```

   It refuses to continue if a row has no image, an image has no row, or the
   folder label disagrees with `Hb < 11 g/dL`. On the real download all three
   checks pass on all 710 images.

The resulting `metadata.csv` has **at least** these columns:

   | column       | meaning                                   |
   |--------------|-------------------------------------------|
   | `image_id`   | filename relative to `data_dir`           |
   | `patient_id` | one id per child — see the caveat below   |
   | `site`       | collection facility (the confounder axis) |
   | `label`      | 1 = anemic (Hb < 11), 0 = not             |
   | `hemoglobin` | g/dL (optional, enables the Hb-regression variant) |
   | `age_months` | optional                                  |
   | `sex`        | optional                                  |

### Two things the adapter cannot fix

**There is no patient identifier.** The spreadsheet has one row per image and
the filenames are a flat sequence, so `patient_id = image_id` — i.e. one image
per child is *assumed*. If the source study photographed both eyes of ~355
children, the patient-level split is leaky and only the leave-one-site-out
results stand. The assumption is recorded in `data/cp-anemic/PREPARED.json`;
verify it against the publication before quoting a patient-level number.

**The images are pre-segmented conjunctiva strips on black**, where the
background is 52-92% of the frame and varies 40 points between images. Colour
features are therefore taken over `roi_mask(img)`; unmasked they would encode
how much black each crop happens to contain instead of the pallor.

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
