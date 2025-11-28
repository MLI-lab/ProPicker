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
4. Install the package:
   ```bash
   pip install .
   ```

## Workflow (step-by-step commands)
1) Prompt selection (GUI):
   ```bash
   propicker-prompt-selector --tomo path/to/volume.mrc --output picks.tsv --output-dir prompt_outputs
   ```
   - Pick points, save TSV and subtomos.
   - Also writes `prompt_coordinates.txt` in `--output-dir` with prompt names (`prompt_1`, …) and coordinates.
   - Display options: invert contrast, display smoothing (σ).

2) Predict locmaps:
   ```bash
   propicker-predict-locmap --tomo tomo1.mrc tomo2.mrc --prompt-subtomos prompt_subtomos --output-dir pred_locmaps
   ```
   - Generates `<tomo>_pred_locmaps.pt` with per-prompt locmaps.

3) Threshold tuning (GUI):
   ```bash
   propicker-pick-from-locmap-gui --locmap pred_locmaps/tomo_pred_locmaps.pt --prompt prompt_1 --tomo tomo1.mrc --output-picks-file picks.tsv
   ```
   - Adjust binarization, size thresholds (min/max enable flags), particle diameter (point size), display smoothing, alpha.
   - Save picks (TSV) and thresholds (JSON).

4) Batch pick from locmaps with saved thresholds:
   ```bash
   propicker-pick-from-locmap --pred-locmap pred_locmaps/tomo_pred_locmaps.pt --prompt prompt_1 --thresholds-json thresholds.json --output-dir pred_picks
   ```

## VNC / Remote GUI
Use this when running on a remote server or in a container without a desktop. Both GUIs accept `--vnc` to auto-start a virtual display and VNC server (localhost only):
- Flags: `--vnc --vnc-port 5901 --vnc-display :1 --vnc-password <pw>` (password required for safety)
- Prompt GUI example:
  ```bash
  propicker-prompt-selector --tomo ... --output picks.tsv --vnc --vnc-password abc
  ```
- Locmap GUI example:
  ```bash
  propicker-pick-from-locmap-gui --locmap ...pt --prompt prompt_1 --tomo ...mrc --output-picks-file picks.tsv --vnc --vnc-password abc
  ```
- Install `xvfb` and `x11vnc` first. Forward the VNC port (default 5901) and connect with a VNC client to `localhost:<port>` using the password you set.
