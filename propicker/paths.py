import os
from pathlib import Path


def _env_or_default(env_var: str, default: str) -> str:
    val = os.environ.get(env_var)
    return val if val else default


# Base dir (not used for shipped weights anymore, but kept for dataset defaults)
PROPICKER_REPO_DIR = Path(__file__).resolve().parent.parent

# Model files: prefer environment overrides; otherwise expect files in CWD (or user-specified paths)
PROPICKER_MODEL_FILE = _env_or_default("PROPICKER_MODEL_FILE", "propicker.ckpt")
TOMOTWIN_MODEL_FILE = _env_or_default("TOMOTWIN_MODEL_FILE", "tomotwin.pth")

# Dataset base dir (env override supported)
DATASETS_DIR = _env_or_default("PROPICKER_DATASETS_DIR", "datasets")

# Dataset paths (may need env overrides if moved)
EMPIAR10988_BASE_DIR = f"{DATASETS_DIR}/empiar/10988/DEF"
TOMOTWIN_TOMO_BASE_DIR = f"{DATASETS_DIR}/tomotwin_data/tomograms"
SHREC2021_BASE_DIR = f"{DATASETS_DIR}/shrec2021/full_dataset"
