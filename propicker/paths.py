import os
from pathlib import Path

# Base dir (not used for shipped weights anymore, but kept for dataset defaults)
PROPICKER_REPO_DIR = Path(__file__).resolve().parent.parent

# Model files: resolved via environment variables (no package defaults)
PROPICKER_MODEL_FILE = os.environ.get("PROPICKER_MODEL_FILE")
TOMOTWIN_MODEL_FILE = os.environ.get("TOMOTWIN_MODEL_FILE")

# Dataset base dir (env override supported; default to relative datasets/)
DATASETS_DIR = os.environ.get("PROPICKER_DATASETS_DIR", "datasets")

# Dataset paths (may need env overrides if moved)
EMPIAR10988_BASE_DIR = f"{DATASETS_DIR}/empiar/10988/DEF"
TOMOTWIN_TOMO_BASE_DIR = f"{DATASETS_DIR}/tomotwin_data/tomograms"
SHREC2021_BASE_DIR = f"{DATASETS_DIR}/shrec2021/full_dataset"
