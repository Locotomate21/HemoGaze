"""Model definition.

"Modern architecture" here means a modern *backbone* used the way a ~700-image
dataset actually rewards: transfer learning plus regularisation, not a
from-scratch Vision Transformer that needs a million images to shine.
ConvNeXt-Tiny is a 2020s convnet that matches ViT accuracy while training
happily on small data; EfficientNet-B0 is a leaner fallback. The honest
experiment is to try both and expect a near-tie at this scale -- and to compare
both against the colour logistic regression in ``baselines.py`` before quoting
either.

Like ``dataset.py``, this module imports torch at module level and is never
imported by the package ``__init__``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_model(backbone: str = "convnext_tiny", pretrained: bool = True,
                dropout: float = 0.3, num_classes: int = 1) -> nn.Module:
    """Binary head by default (num_classes=1 -> single logit + BCEWithLogits).

    Set num_classes>1 for the multi-grade variant. The regression variant also
    uses a single output, so it shares this path -- only the loss changes.
    """
    import timm

    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )


def build_regression_model(backbone: str = "convnext_tiny",
                           pretrained: bool = True,
                           dropout: float = 0.3) -> nn.Module:
    """Single continuous output for predicting hemoglobin (g/dL)."""
    return build_model(backbone, pretrained, dropout, num_classes=1)


def freeze_backbone(model: nn.Module, freeze: bool = True) -> nn.Module:
    """Freeze everything but the classifier head.

    With a few hundred images, fine-tuning 28M parameters from epoch 0 mostly
    memorises the training facilities. A short head-only warmup lets the new head
    stop producing noise before the pretrained features are allowed to move.
    """
    head = model.get_classifier()
    for p in model.parameters():
        p.requires_grad = not freeze
    for p in head.parameters():
        p.requires_grad = True
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    """Reported per run: with ~700 images, the parameter count is context the
    reader needs in order to judge the result."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last spatial conv layer, used as the Grad-CAM target.

    Grad-CAM matters here for a specific reason: it is how we check whether the
    model looks at the conjunctiva or at an eyelash, a flash reflection, or the
    ruler someone held in frame at one facility.
    """
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError(f"No Conv2d layer found in {type(model).__name__}; "
                         f"pick a target layer manually for this backbone.")
    return convs[-1]


def load_weights(model: nn.Module, path, device: str = "cpu") -> nn.Module:
    """Load a checkpoint saved by ``scripts/02_train.py``."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model
