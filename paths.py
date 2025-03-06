import os

# path to base directory of this repo
PROPICKER_REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# path to the ProPicker model file downloaded from Google Drive
PROPICKER_MODEL_FILE = f"{PROPICKER_REPO_DIR}/deepetpicker.ckpt"
# path to TomoTwin model file downloaded with .sh script
TOMOTWIN_MODEL_FILE = f"{PROPICKER_REPO_DIR}/tomotwin.pth"

# path to the directory containing the datasets, modify this if you want to store the datasets elsewhere
DATASETS_DIR = f"{PROPICKER_REPO_DIR}/datasets"

# empiar10988 dataset, downloaded with .sh script
EMPIAR10988_BASE_DIR = f"{DATASETS_DIR}/empiar/10988/DEF"

# training datasets, downloaded with .sh script
TOMOTWIN_TOMO_BASE_DIR = f"{DATASETS_DIR}/tomotwin_data/tomograms"
SHREC2021_BASE_DIR = f"{DATASETS_DIR}/shrec2021/full_dataset"

