"""FastAPI service: conjunctiva photograph in, hemoglobin estimate out.

The model served here is the ConvNeXt-Tiny hemoglobin regressor from
``training/``. It works mechanically and it is **not fit for clinical use**, and
this service says so in every response rather than in a footnote someone can
strip. The measured reasons, from the project's own experiments:

* MAE 1.68 g/dL against 1.80 for predicting the training mean -- a 7%
  improvement over guessing the average.
* Bland-Altman limits of agreement near +/-4 g/dL on a scale that runs 3 to 17.
* On an external dataset of Italian and Indian adults the correlation between
  true and predicted hemoglobin is **negative** (Spearman -0.28 overall, -0.53
  in Italy). Outside the training population the model is confidently wrong.

So `valid_for_screening` is hard-coded false and the payload carries the numbers
that justify it. A deployment that wants to drop the warning has to edit code,
which is the point.

Why the endpoint returns g/dL rather than a label: hemoglobin is
population-independent, so the caller applies the WHO cutoff appropriate to the
patient (11 g/dL for children 6-59 months, 12 for adult women, 13 for adult men).
A server that returned "anemic: true" would be baking in a threshold it cannot
know is right for the person in the photograph.

Run:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "processing"))
sys.path.insert(0, str(ROOT / "training"))

from features import roi_mask  # noqa: E402

# Set by the deployment; see container_images/hemogaze/README.md.
WEIGHTS = Path(__file__).resolve().parent / "model.pt"
CONFIG = Path(__file__).resolve().parent / "config.yaml"

# Measured on CP-AnemiC and Eyes-Defy-Anemia. See the project README.
EVIDENCE = {
    "mae_g_dl": 1.68,
    "mae_predicting_the_training_mean_g_dl": 1.80,
    "bland_altman_limits_of_agreement_g_dl": [-4.0, 4.0],
    "external_spearman_italy": -0.53,
    "training_population": "children 6-59 months, ten hospitals in Ghana",
}
WARNING = (
    "NOT FOR CLINICAL USE. This estimate is 7% better than predicting the "
    "population average, and on adults outside the training population the "
    "model's error correlates negatively with true hemoglobin -- it ranks the "
    "healthy as anemic. Lab hemoglobin is the reference standard."
)

app = FastAPI(
    title="HemoGaze hemoglobin estimator",
    description=__doc__,
    version="1.0.0",
)
_model = None


class Prediction(BaseModel):
    hemoglobin_g_dl: float = Field(..., description="Predicted haemoglobin")
    roi_fraction: float = Field(
        ..., description="Share of the frame identified as conjunctiva. Very low "
                         "values mean the crop is mostly background and the "
                         "estimate is unreliable.")
    cutoff_g_dl: float | None = None
    below_cutoff: bool | None = Field(
        None, description="Only present when the caller supplies a cutoff. The "
                          "server does not choose one: the WHO threshold depends "
                          "on age, sex and pregnancy.")
    valid_for_screening: bool = Field(
        False, description="Always false. See warning and evidence.")
    warning: str = WARNING
    evidence: dict = EVIDENCE


def load_model():
    """Lazy, so the container starts and answers /health even without weights."""
    global _model
    if _model is not None:
        return _model
    import torch

    from config import load_config
    from model import build_model, load_weights
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"No weights at {WEIGHTS}. Fetch them with "
            f"'dvc pull reports' and copy the regression run's model.pt here.")
    cfg = load_config(CONFIG) if CONFIG.exists() else None
    backbone = cfg.backbone if cfg else "convnext_tiny"
    dropout = cfg.dropout if cfg else 0.3
    m = build_model(backbone, pretrained=False, dropout=dropout)
    load_weights(m, WEIGHTS, "cpu")
    m.eval()
    _model = (m, cfg.image_size if cfg else 128,
              cfg.roi_black_threshold if cfg else 20)
    torch.set_grad_enabled(False)
    return _model


@app.get("/health")
def health() -> dict:
    """Liveness plus whether weights are actually present, because a service
    that answers 200 while unable to predict is worse than one that is down."""
    return {"status": "ok", "weights_present": WEIGHTS.exists(),
            "valid_for_screening": False}


@app.post("/predict", response_model=Prediction)
async def predict(
    image: UploadFile = File(..., description="Palpebral conjunctiva crop"),
    cutoff_g_dl: float | None = Query(
        None, description="Optional WHO threshold to compare against: 11 for "
                          "children 6-59 months, 12 adult women, 13 adult men."),
) -> Prediction:
    import torch
    from torchvision import transforms

    from dataset import IMAGENET_MEAN, IMAGENET_STD

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "Empty upload.")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Not a readable image: {exc}") from exc

    try:
        model, size, threshold = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc

    arr = np.asarray(img)
    roi = float(roi_mask(arr, threshold).mean())

    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    hb = float(model(tf(img).unsqueeze(0)).squeeze())

    return Prediction(
        hemoglobin_g_dl=round(hb, 2),
        roi_fraction=round(roi, 3),
        cutoff_g_dl=cutoff_g_dl,
        below_cutoff=None if cutoff_g_dl is None else bool(hb < cutoff_g_dl),
    )
