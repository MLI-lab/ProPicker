# ProPicker GUI Quickstart

## Overview
The prompt-based picking workflow with ProPicker involves four main steps:
1) Pick prompts in one tomogram with `propicker-prompt-selector`.
2) Segment tomograms into locmaps with `propicker-predict-locmap` using those prompts.
3) Tune thresholds to turn locmaps into picks with `propicker-pick-from-locmap-gui`.
4) Generate picks for all tomograms in batch with `propicker-pick-from-locmap` using the saved thresholds.

## Install
1. Create/activate a conda env (recommended):
   ```bash
   conda create -n propicker python=3.11 -y
   conda activate propicker
   ```
2. Install system deps for Qt/napari (Ubuntu/Debian) so GUIs render correctly:
   ```bash
   sudo apt-get update && sudo apt-get install -y \
     libgl1 libegl1 libopengl0 libgl1-mesa-dri libgl1-mesa-glx \
     libglib2.0-0 libxcb-cursor0 libxkbcommon-x11-0 libfontconfig1 \
     libdbus-1-3 libx11-xcb1 libxrender1 libsm6 libice6 libxext6 \
     libxi6 libxcomposite1 libxcursor1 libxtst6 libxrandr2 libxdamage1 libxss1
   ```
3. (Optional for headless/remote) install:
   ```bash
   sudo apt-get install -y xvfb x11vnc
   ```
4. Install the package (set env vars for model paths or pass CLI flags):
   ```bash
   pip install .
   ```

Model/data paths (must be provided via env or flags):
- Set env vars as part of setup:
  ```bash
  export PROPICKER_MODEL_FILE=/abs/path/to/propicker.ckpt
  export TOMOTWIN_MODEL_FILE=/abs/path/to/tomotwin.pth
  export PROPICKER_DATASETS_DIR=/abs/path/to/datasets
  ```
- Or pass model paths via CLI flags (e.g., `--propicker-ckpt`, `--tomotwin-ckpt`). Defaults are not bundled.

## Workflow (step-by-step commands)
1) Prompt selection (GUI):
   ```bash
   propicker-prompt-selector --tomo path/to/volume.mrc --output-dir prompt_outputs
   ```
   - Pick one or more prompts; saves `prompt_coords.tsv`, `prompt_coordinates.txt`, and prompt subtomos inside `--output-dir`.
   - **IMPORTANT:** ProPicker requires bright particles on a dark background. If your tomogram has dark particles on a bright background, you can fix this by toggling the "invert contrast" option in the GUI.
   - Display options: display smoothing (σ).

2) Predict locmaps:
   ```bash
   propicker-predict-locmap --tomo tomo1.mrc tomo2.mrc --prompt-subtomos prompt_subtomos --output-dir pred_locmaps
   ```
   - Generates `<tomo>_pred_locmaps.pt` with per-prompt locmaps.
   - **IMPORTANT:** Use `--invert-contrast` if necessary to ensure bright particles on dark background (as in Step 1).

3) Threshold tuning (GUI):
   ```bash
   propicker-pick-from-locmap-gui --locmap pred_locmaps/tomo_pred_locmaps.pt --prompt prompt_1 --tomo tomo1.mrc --output-dir picks
   ```
   - Adjust binarization, size thresholds (min/max enable flags), particle diameter (used to compute a particle volume)
   - Prompt cannot be changed in the gui; must restart GUI to switch prompts
   - Size thresholds = particle volume (from diameter) × min/max multipliers; toggles control whether min/max are applied.
   - Saves picks (`<locmap_name>_picks.tsv`) and thresholds (`<locmap_name>_thresholds.json`) in `--output-dir`.

4) Batch pick from locmaps with saved thresholds:
   ```bash
   propicker-pick-from-locmap --pred-locmap pred_locmaps/tomo_pred_locmaps.pt --prompt prompt_1 --thresholds-json thresholds.json --output-dir pred_picks
   ```

## VNC / Remote GUI
Use this when running on a remote server or in a container without a desktop. Both GUIs accept `--vnc` to auto-start a virtual display and VNC server (localhost only):
- Flags: `--vnc --vnc-port 5901 --vnc-display :1 --vnc-password <pw>` (password required for safety)
- Prompt GUI example:
  ```bash
  propicker-prompt-selector --tomo ... --output-dir prompt_outputs --vnc --vnc-password abc
  ```
- Locmap GUI example:
  ```bash
  propicker-pick-from-locmap-gui --locmap ...pt --prompt prompt_1 --tomo ...mrc --output-dir picks --vnc --vnc-password abc
  ```
- Install `xvfb` and `x11vnc` first. Forward the VNC port (default 5901) and connect with a VNC client to `localhost:<port>` using the password you set.
