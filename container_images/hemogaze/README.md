# container_images/hemogaze

The **production** image. Only what has already been tested in
`Python/hemogaze/Dockerfile` reaches this one.

    docker compose up          # from the repo root

## What it deliberately does not contain

| left out | why |
|---|---|
| `training/` (except `dataset.py`, `model.py`) | a serving container that can retrain itself is attack surface for no benefit |
| `tests/`, `dvc`, `ruff` | development tooling |
| GPU torch | the CPU wheels are ~2 GB smaller and this is one 128x128 forward pass per request |
| `matplotlib`, `openpyxl` | figures and spreadsheet parsing are training-time concerns |
| **the model weights** | see below |

## Weights are mounted, not baked in

They live in DVC/DagsHub. `docker-compose.yml` mounts them read-only from
`reports/runs_hb/hb128_patient_level/`, so:

- a 107 MB binary never enters the registry,
- shipping a retrained model is a restart rather than a rebuild,
- the image is identical whichever model it serves, which makes it reproducible.

Fetch them first:

    dvc pull reports

Without them the container still starts and `/health` answers with
`weights_present: false`; `/predict` returns 503. That is intentional -- a
service that answers 200 while unable to predict is worse than one that is down.

## Security

Runs as uid 10001, not root. The healthcheck hits `/health`, which reports
whether weights are actually loaded, so an orchestrator can distinguish "process
alive" from "able to serve".

## What it serves

A model that is **not fit for clinical use**, and which says so in every
response. See `Python/hemogaze/server/README.md` and the project README for the
measurements behind that.
