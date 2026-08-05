"""Tests for the serving layer.

These run without weights on disk, which is the point: the service has to behave
correctly when it *cannot* predict, and that path is the one most likely to be
wrong in production.

The strongest assertions here are not about accuracy. They are about the warning:
a screening service whose caveat can be dropped by a client, or which reports a
number as clinically usable, is worse than one that is down.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "processing", "training", "server"):
    sys.path.insert(0, str(ROOT / sub) if sub else str(ROOT))

fastapi = pytest.importorskip("fastapi", reason="serving extras not installed")
pytest.importorskip("httpx", reason="TestClient needs httpx")
from fastapi.testclient import TestClient  # noqa: E402

from server.app import WARNING, app  # noqa: E402

client = TestClient(app)


def _png(width=60, height=30, colour=(180, 95, 95)) -> bytes:
    """A crude conjunctiva stand-in: a coloured band on black, like the real
    pre-segmented crops."""
    a = np.zeros((height, width, 3), dtype="uint8")
    a[height // 3: 2 * height // 3, :] = colour
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def test_health_reports_whether_it_can_actually_predict():
    """A service that answers 200 while unable to serve is worse than one that
    is down, so /health separates 'process alive' from 'weights present'."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "weights_present" in body
    assert body["valid_for_screening"] is False


def test_predict_without_weights_fails_loudly():
    """503, not a fabricated number and not a 200 with nulls."""
    from server import app as mod
    if mod.WEIGHTS.exists():
        pytest.skip("weights are installed; the no-weights path cannot be tested")
    r = client.post("/predict", files={"image": ("x.png", _png(), "image/png")})
    assert r.status_code == 503
    assert "weights" in r.json()["detail"].lower()


def test_rejects_a_non_image_upload():
    r = client.post("/predict",
                    files={"image": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_rejects_an_empty_upload():
    r = client.post("/predict", files={"image": ("x.png", b"", "image/png")})
    assert r.status_code == 400


def test_the_warning_is_not_empty_and_says_what_matters():
    """The caveat is part of the contract, so it is asserted like any other
    field. If someone softens it, a test fails."""
    assert "NOT FOR CLINICAL USE" in WARNING
    assert "reference standard" in WARNING.lower()


def test_response_schema_pins_valid_for_screening_to_false():
    """Not a default a caller can flip: the field is declared false and the
    evidence travels with it."""
    from server.app import EVIDENCE, Prediction
    p = Prediction(hemoglobin_g_dl=10.0, roi_fraction=0.3)
    assert p.valid_for_screening is False
    assert p.warning == WARNING
    assert p.evidence["mae_g_dl"] < p.evidence["mae_predicting_the_training_mean_g_dl"]
    assert EVIDENCE["external_spearman_italy"] < 0    # the inversion, on record


def test_no_cutoff_means_no_verdict():
    """The server must not invent a threshold: WHO cutoffs depend on age, sex
    and pregnancy, none of which it can see."""
    from server.app import Prediction
    assert Prediction(hemoglobin_g_dl=11.5, roi_fraction=0.3).below_cutoff is None
    p = Prediction(hemoglobin_g_dl=11.5, roi_fraction=0.3,
                   cutoff_g_dl=12.0, below_cutoff=True)
    assert p.below_cutoff is True
