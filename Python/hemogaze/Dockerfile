# Development image: this is where things get tested.
#
# It carries the full stack -- torch, timm, pytest, the training pipelines and
# the datasets mounted from DVC -- so the whole chain can be exercised inside a
# container that matches CI. Nothing here is meant to ship.
#
# The production counterpart lives in container_images/hemogaze/Dockerfile and
# only receives what has already passed through this one.
FROM python:3.13-slim

# libgl/libglib: Pillow and torchvision need them for image decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing source does not invalidate the wheel cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir dvc[http] ruff

# WORKDIR is the package root, which is why the modules import as
# `from features import ...` with no packaging step.
COPY processing/ processing/
COPY training/   training/
COPY server/     server/
COPY examples/   examples/
COPY tests/      tests/

ENV PYTHONPATH=/app:/app/processing:/app/training \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Fail the build if the honest core is broken. These 47 tests need no data and
# no GPU, so there is no excuse for shipping past them.
RUN python -m pytest tests -q

CMD ["python", "-m", "pytest", "tests", "-q"]
