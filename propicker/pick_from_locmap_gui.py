#!/usr/bin/env python
"""
Napari GUI to turn predicted locmaps into picks via clustering.

Usage example:
    python locmap_to_picks_gui.py --locmap pred_locmaps/sample_pred_locmaps.pt --prompt prompt_1 --output picks.tsv
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import mrcfile
import napari
import numpy as np
import torch
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import gaussian_filter

from propicker.clustering_and_picking import get_cluster_centroids_df
from propicker.utils.mrctools import load_mrc_data
from scipy.ndimage import gaussian_filter

# Prefer PySide6 and software rendering by default (matches prompt_picker_gui).
os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")


def load_locmap(path: Path, prompt: str | None = None) -> Tuple[np.ndarray, str]:
    """Load a locmap volume from a .pt dict or .mrc file."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Locmap file not found: {path}")

    if path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        if not isinstance(data, dict):
            raise ValueError(f"Expected a dict in {path}, got {type(data)}")
        keys = list(data.keys())
        if prompt is None:
            prompt = keys[0]
            print(f"No prompt provided; using first key: {prompt}")
        if prompt not in data:
            raise KeyError(f"Prompt '{prompt}' not found in {path}. Available: {keys}")
        vol = data[prompt]
        if torch.is_tensor(vol):
            vol = vol.cpu().numpy()
        locmap = np.asarray(vol)
        name = prompt
    else:
        with mrcfile.open(path, permissive=True) as fh:
            locmap = np.asarray(fh.data)
        name = path.stem
    if locmap.ndim != 3:
        raise ValueError(f"Locmap must be 3D, got shape {locmap.shape}")
    return locmap.astype(np.float32, copy=False), name


def particle_volume_from_diameter(diameter: float) -> float:
    """Compute voxel volume for a spherical particle diameter."""
    radius = diameter / 2.0
    return 4.0 / 3.0 * np.pi * (radius ** 3)


def start_vnc(display: str, port: int, password: str) -> None:
    """Launch Xvfb and x11vnc for headless viewing."""
    if shutil.which("Xvfb") is None or shutil.which("x11vnc") is None:
        raise SystemExit("xvfb and x11vnc are required for --vnc. Install them and retry.")
    if not password:
        raise SystemExit("--vnc-password is required when using --vnc (for safety).")

    xvfb_cmd = ["Xvfb", display, "-screen", "0", "1920x1080x24"]
    vnc_cmd = [
        "x11vnc",
        "-display",
        display,
        "-localhost",
        "-shared",
        "-forever",
        "-rfbport",
        str(port),
        "-passwd",
        password,
    ]
    xvfb_proc = subprocess.Popen(xvfb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    vnc_proc = subprocess.Popen(vnc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _cleanup():
        for proc in (vnc_proc, xvfb_proc):
            if proc.poll() is None:
                proc.terminate()

    atexit.register(_cleanup)

def build_panel(
    run_clustering,
    save_picks,
    save_thresholds,
    bin_thresh_spin: QDoubleSpinBox,
    min_mult_spin: QDoubleSpinBox,
    max_mult_spin: QDoubleSpinBox,
    min_checkbox: QCheckBox,
    max_checkbox: QCheckBox,
    diameter_spin: QDoubleSpinBox,
    smooth_spin: QDoubleSpinBox,
    alpha_spin: QDoubleSpinBox,
    refresh_overlay,
    apply_filters,
    status_label: QLabel,
    run_button: QPushButton,
) -> QWidget:
    panel = QWidget()
    layout = QVBoxLayout()

    info = QLabel(
        "Workflow:\n"
        " 1) Adjust binarization threshold.\n"
        " 2) Run clustering (creates picks from threshold).\n"
        " 3) Tune size thresholds (changes auto-apply).\n"
        " 4) Save picks and/or thresholds to disk."
    )
    info.setWordWrap(True)

    form = QFormLayout()
    form.addRow("Binarization threshold", bin_thresh_spin)
    form.addRow(min_checkbox)
    form.addRow("Min size × volume", min_mult_spin)
    form.addRow(max_checkbox)
    form.addRow("Max size × volume", max_mult_spin)
    form.addRow("Particle diameter (voxels)", diameter_spin)
    form.addRow("Display smoothing (σ)", smooth_spin)
    form.addRow("Locmap alpha", alpha_spin)

    btn_run = run_button
    bin_thresh_spin.valueChanged.connect(lambda v: refresh_overlay())
    min_mult_spin.valueChanged.connect(lambda v: apply_filters())
    max_mult_spin.valueChanged.connect(lambda v: apply_filters())
    min_checkbox.toggled.connect(lambda _: apply_filters())
    max_checkbox.toggled.connect(lambda _: apply_filters())
    btn_save = QPushButton("Save picks")
    btn_save.clicked.connect(save_picks)
    btn_save_thresh = QPushButton("Save thresholds")
    btn_save_thresh.clicked.connect(save_thresholds)

    layout.addWidget(info)
    layout.addLayout(form)
    layout.addWidget(btn_run)
    layout.addWidget(btn_save)
    layout.addWidget(btn_save_thresh)
    layout.addWidget(status_label)
    panel.setLayout(layout)
    return panel


def save_picks_tsv(coords_zyx: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "\t".join(["X", "Y", "Z"])
    np.savetxt(output_path, coords_zyx[:, ::-1], fmt="%.3f", delimiter="\t", header=header, comments="")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Napari GUI to convert locmaps to picks via clustering.")
    parser.add_argument("--locmap", required=True, type=Path, help="Locmap file (.pt dict or .mrc).")
    parser.add_argument("--tomo", type=Path, default=None, help="Optional tomogram to show underneath the locmap.")
    parser.add_argument("--prompt", default=None, help="Prompt key to load from a .pt file (default: first key).")
    parser.add_argument("--output-dir", type=Path, default=Path("picks"), help="Directory to save picks and thresholds. Created if needed.")
    parser.add_argument("--vnc", action="store_true", help="Run with an embedded Xvfb + x11vnc for remote viewing.")
    parser.add_argument("--vnc-port", type=int, default=5901, help="VNC port (default: 5901).")
    parser.add_argument("--vnc-display", type=str, default=":1", help="X display to use for VNC (default: :1).")
    parser.add_argument("--vnc-password", type=str, default=None, help="Password for VNC (required if --vnc).")
    args = parser.parse_args(argv)

    if args.vnc:
        start_vnc(args.vnc_display, args.vnc_port, password=args.vnc_password)
        os.environ["DISPLAY"] = args.vnc_display
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        print(f"VNC running on localhost:{args.vnc_port} (DISPLAY {args.vnc_display}). Connect with your VNC client.")

    locmap, locmap_name = load_locmap(args.locmap, prompt=args.prompt)
    locmap_data = locmap.copy()
    binary_mask = np.zeros_like(locmap, dtype=bool)
    locmap_display = locmap_data

    viewer = napari.Viewer(title=f"Locmap to picks - {locmap_name}")
    tomo_layer = None
    tomo_vol = None
    if args.tomo is not None:
        tomo_vol = load_mrc_data(args.tomo).float().cpu().numpy()
        if tomo_vol.ndim != 3:
            raise ValueError(f"Tomo must be 3D, got shape {tomo_vol.shape}")
        p01, p99 = np.percentile(tomo_vol, [1, 99])
        tomo_layer = viewer.add_image(
            tomo_vol,
            name=args.tomo.stem,
            colormap="gray",
            blending="additive",
            contrast_limits=(float(p01), float(p99)),
        )
    locmap_layer = viewer.add_image(
        (locmap > 0.5).astype(np.float32),
        name=f"{locmap_name}_locmap",
        colormap="magma",
        blending="additive",
        opacity=0.5,
    )
    base_kwargs = dict(
        face_color="transparent",
        border_color="red",
        border_width=0.75,
        border_width_is_relative=False,
        size=25.0,
        out_of_slice_display=True,
        opacity=1.0,
    )
    all_points = viewer.add_points(np.empty((0, 3)), name="all_clusters", **base_kwargs, visible=False)
    filtered_points = viewer.add_points(np.empty((0, 3)), name="filtered_clusters", **base_kwargs)

    bin_thresh_spin = QDoubleSpinBox()
    bin_thresh_spin.setRange(0.0, 1.0)
    bin_thresh_spin.setSingleStep(0.01)
    bin_thresh_spin.setValue(0.5)

    min_mult_spin = QDoubleSpinBox()
    min_mult_spin.setRange(0.0, 10.0)
    min_mult_spin.setSingleStep(0.05)
    min_mult_spin.setValue(0.1)

    max_mult_spin = QDoubleSpinBox()
    max_mult_spin.setRange(0.0, 10.0)
    max_mult_spin.setSingleStep(0.05)
    max_mult_spin.setValue(1.2)

    diameter_spin = QDoubleSpinBox()
    diameter_spin.setRange(1.0, 500.0)
    diameter_spin.setSingleStep(1.0)
    diameter_spin.setValue(25.0)
    smooth_spin = QDoubleSpinBox()
    smooth_spin.setRange(0.0, 10.0)
    smooth_spin.setSingleStep(0.2)
    smooth_spin.setValue(0.0)
    alpha_spin = QDoubleSpinBox()
    alpha_spin.setRange(0.0, 1.0)
    alpha_spin.setSingleStep(0.05)
    alpha_spin.setValue(0.5)
    min_checkbox = QCheckBox("Apply min size threshold")
    min_checkbox.setChecked(True)
    max_checkbox = QCheckBox("Apply max size threshold")
    max_checkbox.setChecked(True)
    status_label = QLabel("")

    state: Dict[str, np.ndarray] = {"all_df": None, "filtered_df": None}

    def smooth_for_display(arr: np.ndarray) -> np.ndarray:
        sigma = smooth_spin.value()
        return gaussian_filter(arr, sigma) if sigma > 0 else arr

    def current_volume() -> float:
        try:
            return float(particle_volume_from_diameter(diameter_spin.value()))
        except Exception:
            return 0.0

    run_button = QPushButton("Run clustering")
    run_button.setEnabled(True)

    def run_clustering(event=None) -> None:
        run_button.setEnabled(False)
        thresh = bin_thresh_spin.value()
        status_label.setText(f"Running clustering at threshold {thresh:.3f}... please wait.")
        viewer.status = status_label.text()
        thresh = bin_thresh_spin.value()
        binary_mask[:] = locmap > thresh
        df = get_cluster_centroids_df(binary_mask, min_cluster_size=1, max_cluster_size=np.inf)
        coords = df[["Z", "Y", "X"]].to_numpy(dtype=float) if len(df) else np.empty((0, 3))
        all_points.data = coords
        state["all_df"] = df
        state["filtered_df"] = None
        apply_filters()
        status_label.setText(f"Clustering complete: {len(df)} clusters at threshold {thresh:.3f}")
        viewer.status = status_label.text()

    def threshold_clusters(event=None) -> None:
        if state["all_df"] is None:
            viewer.status = "Run clustering first."
            return
        vol = current_volume()
        if vol <= 0:
            viewer.status = "Invalid particle diameter; enter a positive number."
            return
        min_size = min_mult_spin.value() * vol if min_checkbox.isChecked() else 0.0
        max_size = max_mult_spin.value() * vol if max_checkbox.isChecked() else np.inf
        if max_size < min_size:
            viewer.status = "Max size must be >= min size."
            return
        df = state["all_df"]
        df_filt = df[(df["size"] >= min_size) & (df["size"] <= max_size)]
        coords = df_filt[["Z", "Y", "X"]].to_numpy(dtype=float) if len(df_filt) else np.empty((0, 3))
        filtered_points.data = coords
        state["filtered_df"] = df_filt
        viewer.status = f"Filtered to {len(df_filt)} clusters (size {min_size:.1f}-{max_size:.1f} voxels)"
    apply_filters = threshold_clusters  # alias for clarity

    def save_picks(event=None) -> None:
        df = state["filtered_df"] if state["filtered_df"] is not None else state["all_df"]
        if df is None or len(df) == 0:
            viewer.status = "No picks to save."
            return
        coords = df[["Z", "Y", "X"]].to_numpy(dtype=float)
        output_picks_file = args.output_dir / f"{locmap_name}_picks.tsv"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_picks_tsv(coords, output_picks_file)
        viewer.status = f"Saved {len(df)} picks to {output_picks_file}"
        print(viewer.status)

    def refresh_overlay() -> None:
        """Update locmap overlay to reflect current binarization threshold."""
        thresh = bin_thresh_spin.value()
        locmap_layer.data = (smooth_for_display(locmap_data) > thresh).astype(np.float32)
        locmap_layer.contrast_limits = (0.0, 1.0)
        viewer.status = f"Overlay thresholded at {thresh:.3f}"

    def save_thresholds(event=None) -> None:
        vol = current_volume()
        min_size = min_mult_spin.value() * vol if min_checkbox.isChecked() else 0.0
        max_size = max_mult_spin.value() * vol if max_checkbox.isChecked() else float("inf")
        data = {
            "prompt": args.prompt,
            "binarization_threshold": bin_thresh_spin.value(),
            "min_size_voxels": min_size,
            "max_size_voxels": max_size,
        }
        out_path = args.output_dir / f"{locmap_name}_thresholds.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        viewer.status = f"Saved thresholds to {out_path}"
        print(viewer.status)

    def update_point_size(val: float) -> None:
        all_points.size = val
        filtered_points.size = val

    def update_display_layers() -> None:
        if tomo_layer is not None and tomo_vol is not None:
            tomo_layer.data = smooth_for_display(tomo_vol)
        refresh_overlay()

    alpha_spin.valueChanged.connect(lambda v: setattr(locmap_layer, "opacity", v))
    diameter_spin.valueChanged.connect(update_point_size)
    smooth_spin.valueChanged.connect(lambda v: update_display_layers())
    bin_thresh_spin.valueChanged.connect(lambda _: run_button.setEnabled(True))
    run_button.clicked.connect(run_clustering)
    update_point_size(diameter_spin.value())
    update_display_layers()
    # Apply initial filter (no picks yet but keeps logic consistent)
    apply_filters()

    panel = build_panel(
        run_clustering=run_clustering,
        save_picks=save_picks,
        bin_thresh_spin=bin_thresh_spin,
        min_mult_spin=min_mult_spin,
        max_mult_spin=max_mult_spin,
        min_checkbox=min_checkbox,
        max_checkbox=max_checkbox,
        diameter_spin=diameter_spin,
        smooth_spin=smooth_spin,
        alpha_spin=alpha_spin,
        refresh_overlay=refresh_overlay,
        apply_filters=apply_filters,
        save_thresholds=save_thresholds,
        status_label=status_label,
        run_button=run_button,
    )
    viewer.window.add_dock_widget(panel, area="right")
    
    viewer.window.resize(1900, 1100)  # tweak as desired
    napari.run()


if __name__ == "__main__":
    main()
