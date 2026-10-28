"""Shared local paths and dataset identifiers for notebooks and the API."""

import os
from pathlib import Path

# Colab Drive is the default; environment overrides make local/VS Code runs possible.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED = Path(os.environ.get("CXR_SHARED_DIR", PROJECT_ROOT / "cxr_shared")).expanduser()
INDIANA_SLUG = "raddar/chest-xrays-indiana-university"
NIH_SLUG = "nih-chest-xrays/sample"

# Point this at an already available dataset. Downloads require explicit opt-in.
DATASET_ROOT = os.environ.get("CXR_DATASET_ROOT")
