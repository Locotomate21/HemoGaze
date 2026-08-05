# server/

FastAPI service around the hemoglobin regressor.

    uvicorn server.app:app --host 0.0.0.0 --port 8000   # from Python/hemogaze/

| endpoint | |
|---|---|
| `GET /health` | liveness, and whether weights are actually present |
| `POST /predict` | multipart `image` -> predicted haemoglobin in g/dL |
| `GET /docs` | OpenAPI UI |

## Two deliberate design choices

**It returns g/dL, not "anemic yes/no".** The WHO cutoff depends on the patient:
11 g/dL for children 6-59 months, 12 for adult women, 13 for adult men. A server
that returned a label would bake in a threshold it cannot know applies to the
person in the photograph. Pass `?cutoff_g_dl=` and the caller decides.

**Every response carries `valid_for_screening: false` and the evidence for it.**
The model is 7% better than predicting the population average, its Bland-Altman
limits of agreement are near +/-4 g/dL on a 3-17 scale, and on Italian adults its
predictions correlate *negatively* with true haemoglobin. Removing that warning
requires editing code, which is intentional.

## Weights

Not in git. Fetch from DVC and place beside the app:

    dvc pull reports
    cp ../../reports/runs_hb/hb128_patient_level/model.pt    server/model.pt
    cp ../../reports/runs_hb/hb128_patient_level/config.yaml server/config.yaml

`/health` reports `weights_present: false` until you do, and `/predict` answers
503 rather than pretending.
