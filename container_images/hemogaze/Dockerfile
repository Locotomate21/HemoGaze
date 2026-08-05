# Production image: only what has already been tested.
#
# Differences from Python/hemogaze/Dockerfile, all deliberate:
#
#   - no training/, no tests/, no dvc, no ruff. A serving container that can
#     retrain itself is a larger attack surface for no benefit.
#   - CPU-only torch, from the PyTorch index. The GPU wheels are ~2 GB and this
#     model runs one 128x128 forward pass per request.
#   - runs as a non-root user.
#   - the model weights are NOT baked in. They live in DVC/DagsHub and are
#     mounted at run time, so the image does not have to be rebuilt to ship a
#     retrained model -- and a 107 MB binary stays out of the registry.
FROM python:3.13-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

# processing/ is needed because the server uses roi_mask and the config loader;
# training/ contributes only dataset.py (normalisation constants) and model.py
# (the architecture to load weights into).
COPY processing/         processing/
COPY training/dataset.py training/model.py training/config.py* training/
COPY server/             server/

ENV PYTHONPATH=/app:/app/processing:/app/training \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 hemogaze && chown -R hemogaze /app
USER hemogaze

EXPOSE 8000

# /health reports weights_present, so an orchestrator can tell "process alive"
# from "actually able to answer".
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
