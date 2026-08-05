# genIA_services

Monorepo for the AI services. This file describes **how the repository works**;
each project documents itself in its own `README.md`.

## Layout

```
Datasets/                     DVC pointers; the data itself lives on DagsHub
  <proyecto>_<tipo>_<version>_<tarea>_<timestamp>.dvc   ← in git, ~150 bytes
  <proyecto>_<tipo>_<version>_<tarea>_<timestamp>/      ← not in git

Python/                       one directory per project, grouped by language
  hemogaze/
    processing/               domain logic
    training/                 training pipelines + configs
    server/                   the service, published over HTTP
    examples/                 example clients: web, CLI, desktop
    tests/
    Dockerfile                ← development: this is where things get tested
    requirements.txt · README.md
    venv/                     ← local only, never committed

container_images/             production images: only what has been tested
  hemogaze/
    Dockerfile · requirements.txt · README.md

docker-compose.yml            orchestration
```

Language folders group by the technology a piece is written in, not by product.
A Go client for a Python service would live under `Go/`.

## The rule that organises everything

> **Code goes to GitHub. Anything that is not code goes to DagsHub, via DVC.**

Images, model weights, prediction dumps, figures: DVC. What git carries is the
`.dvc` pointer, ~150 bytes, so `git clone` stays fast and the history stays
readable.

One refinement we apply deliberately: **small text artefacts that are evidence
stay in git.** Metrics JSON, resolved configs and result tables are a few KB each
and only mean anything next to the commit that produced them. Splitting them
across two systems would mean cross-referencing to answer "what did commit X
score". Binaries and bulk go to DVC.

    dvc pull                  # fetch data and weights
    dvc push                  # after adding or changing them

Credentials go in `.dvc/config.local`, which DVC gitignores. Never in
`.dvc/config`.

## The two Dockerfiles

This is the part that most often confuses newcomers, and it is intentional:

| | where | what it is |
|---|---|---|
| development | `Python/<proyecto>/Dockerfile` | full stack, runs the test suite, builds fail if tests fail |
| production | `container_images/<proyecto>/Dockerfile` | only what has already passed, CPU-only, non-root, no training code |

    docker compose --profile dev up    # test
    docker compose up                  # serve

## Practices

- Tests run without data and without a GPU, so CI is cheap and nobody is blocked.
- The development image runs `pytest` during the build: a broken suite cannot be
  built past.
- Model weights are **mounted**, never baked into an image. Shipping a retrained
  model is a restart, not a rebuild.
- Datasets are mounted read-only in containers: their only other copy is on
  DagsHub.
- No patient data, no credentials and no binaries in git. `.gitignore` denies
  `data/` wholesale and allows back only documentation, so a new dataset dropped
  in is covered without anyone remembering to add it.

## Projects

| project | what it is | status |
|---|---|---|
| [`Python/hemogaze`](Python/hemogaze/) | non-invasive anemia screening from conjunctiva photographs | research complete; **the model is not fit for clinical use and the service says so in every response** |
