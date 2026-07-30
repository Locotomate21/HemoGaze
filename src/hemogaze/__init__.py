"""HemoGaze: non-invasive anemia screening from conjunctiva images,
with honest cross-site validation."""

__version__ = "0.1.0"

# Only the torch-free modules are imported here, so `import hemogaze` works
# with no deep-learning stack installed. `dataset` and `model` are imported
# explicitly by the training scripts.
from . import baselines, config, features, metrics, splits  # noqa: F401
