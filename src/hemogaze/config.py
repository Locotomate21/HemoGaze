"""Typed config, loaded from a YAML file. Every run logs the resolved config
next to its metrics so results are reproducible and auditable."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import yaml


@dataclass
class Config:
    # data
    data_dir: str = "data/cp-anemic"
    metadata_csv: str = "data/cp-anemic/metadata.csv"
    image_size: int = 224
    roi_black_threshold: int = 20   # pixels dimmer than this are background
    # Shortcut removal for the CNN. Off by default so the first experiment stays
    # reproducible; config/cpanemic_roi.yaml turns both on.
    crop_to_roi: bool = False       # crop away the framing before the network sees it
    strong_aug: bool = False        # RandomResizedCrop so the silhouette stops being reliable
    randomise_background: bool = False   # redraw the segmentation background every epoch
    silhouette_only: bool = False   # positive control: train on the ROI shape alone

    # task: "classification" predicts the binary label; "regression" predicts
    # hemoglobin in g/dL. Regression is the more general target -- the WHO
    # cutoff (11 for children, 12/13 for adults) is applied after the model, so
    # one set of predictions serves any population.
    task: str = "classification"
    target_col: str = "hemoglobin"   # used when task == "regression"

    # split
    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.15

    # model  (see model-architect agent for why ConvNeXt-Tiny at this scale)
    backbone: str = "convnext_tiny"     # timm name; efficientnet_b0 is a lean alt
    pretrained: bool = True
    dropout: float = 0.3

    # training
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-2
    early_stop_patience: int = 6
    warmup_epochs: int = 2              # head-only epochs before unfreezing
    num_workers: int = 0                # 0 is the safe default on Windows/spawn
    device: str = "auto"                # "auto" | "cpu" | "cuda"

    # evaluation
    spec_target: float = 0.90           # operating point for sensitivity report
    n_calib_bins: int = 10

    # bookkeeping
    out_dir: str = "reports/runs"
    run_name: str = "baseline"
    baseline_dir: str = "reports/baselines"   # 02_train refuses to run without it

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def is_synthetic(self) -> bool:
        """True when data_dir holds the fabricated stand-in dataset.

        Every script checks this and stamps SYNTHETIC on its output, so a test
        run can never be mistaken for a result later.
        """
        return (Path(self.data_dir) / "SYNTHETIC.json").exists()

    def data_tag(self) -> str:
        return "SYNTHETIC" if self.is_synthetic() else "real"


def load_config(path: str | Path | None) -> Config:
    """Load YAML into the dataclass, rejecting unknown keys.

    Silently ignoring a misspelled key is how you end up believing you trained
    with settings you never actually applied.
    """
    if path is None:
        return Config()
    data = yaml.safe_load(Path(path).read_text()) or {}
    known = set(Config.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}. "
                         f"Known keys: {sorted(known)}")
    return Config(**data)
