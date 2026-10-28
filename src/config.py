"""Shared paths and dataset identifiers used by the Colab notebooks."""

import os
from pathlib import Path

# Colab Drive is the default; environment overrides make local/VS Code runs possible.
SHARED = Path(os.environ.get("CXR_SHARED_DIR", "/content/drive/MyDrive/cxr_shared"))
INDIANA_SLUG = "raddar/chest-xrays-indiana-university"
NIH_SLUG = "nih-chest-xrays/sample"

# Optional explicit dataset directory. If unset, src.data resolves it with KaggleHub.
DATASET_ROOT = os.environ.get("CXR_DATASET_ROOT")
