#!/usr/bin/env python
"""
Simple Napari GUI to pick prompt locations in a tomogram and extract subtomos around them.

Usage example:
    python prompt_selector_gui.py --tomo path/to/volume.mrc --output-dir prompt_outputs [--invert-contrast]

Controls:
  - Scroll mouse wheel / use Z slider to move through slices.
  - Left click to add a point (points are stored in Napari's Z, Y, X order).
  - Select a point and press Delete/Backspace to remove.
  - Press "Extract selected prompts" in the side panel or hit the "S" key to write to disk.
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Tuple

# Prefer PySide6 and a headless-friendly Qt platform by default.
os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")

try:
    import mrcfile
    import napari
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget, QCheckBox, QDoubleSpinBox, QFormLayout
except Exception as exc:  # noqa: BLE001 - show a clear hint for Qt binding issues
    hint = (
        "Failed to import GUI dependencies.\n"
        "If you installed PySide6, ensure qtpy is installed and QT_API=pyside6 "
        "(set automatically by this script).\n"
        "Try: pip install 'napari[pyside6]' mrcfile numpy qtpy\n"
    )
    if "libGL" in str(exc):
        hint += (
            "The error mentions missing libGL; install system OpenGL libs, e.g.:\n"
            "  conda install -c conda-forge libgl1\n"
            "or on apt-based systems: sudo apt-get install -y libgl1 libglib2.0-0\n"
        )
    sys.stderr.write(hint)
    raise


def load_tomogram(path: Path) -> np.ndarray:
    """Load a tomogram as a float32 3D array."""
    if not path.exists():
        raise FileNotFoundError(f"Tomogram not found: {path}")
    with mrcfile.open(path, permissive=True) as fh:
        data = np.asarray(fh.data)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape}")
    return data.astype(np.float32, copy=False)


def percentile_contrast(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> Tuple[float, float]:
    """Compute percentile-based contrast limits."""
    p_low, p_high = np.percentile(arr, [low, high])
    return float(p_low), float(p_high)


def reorder_points(points: np.ndarray, order: str) -> np.ndarray:
    """Reorder points from napari's internal ZYX order to a requested order."""
    if points.size == 0:
        return points
    axis_to_idx = {"z": 0, "y": 1, "x": 2}
    idx = [axis_to_idx[c] for c in order]
    return points[:, idx]


def save_points(points: np.ndarray, output_path: Path, order: str) -> None:
    """Persist picked points to disk as a TSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = reorder_points(points, order)
    header = "\t".join(list(order.upper()))
    np.savetxt(output_path, ordered, fmt="%.3f", delimiter="\t", header=header, comments="")


def extract_subtomo(volume: np.ndarray, center: np.ndarray, size: int = 37) -> np.ndarray:
    """Extract a cubic subtomo around a point, padding with zeros at edges."""
    if size % 2 == 0:
        raise ValueError("Subtomo size must be odd to center on a voxel.")
    half = size // 2
    center = np.round(center).astype(int)
    # Keep center inside volume bounds to avoid empty slices when a point is outside.
    center = np.clip(center, 0, np.array(volume.shape) - 1)
    start = center - half
    stop = center + half + 1  # exclusive

    subtomo = np.zeros((size, size, size), dtype=volume.dtype)

    src_slices = []
    dst_slices = []
    for axis in range(3):
        src_start = max(start[axis], 0)
        src_stop = min(stop[axis], volume.shape[axis])
        if src_stop <= src_start:
            # Point outside bounds on this axis; return zeros.
            return subtomo
        dst_start = src_start - start[axis]
        dst_stop = dst_start + (src_stop - src_start)
        src_slices.append(slice(src_start, src_stop))
        dst_slices.append(slice(dst_start, dst_stop))

    subtomo[tuple(dst_slices)] = volume[tuple(src_slices)]
    return subtomo


def save_subtomos(points: np.ndarray, volume: np.ndarray, dest_dir: Path, size: int = 37, invert_factor: int = 1) -> None:
    """Extract and save subtomos for each prompt point."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for idx, center in enumerate(points, start=1):
        cube = extract_subtomo(volume, center, size=size) * invert_factor
        name = dest_dir / f"prompt_{idx}.mrc"
        with mrcfile.new(name, overwrite=True) as mrc:
            mrc.set_data(cube.astype(np.float32, copy=False))


def build_panel(on_save, invert_checkbox: QCheckBox, smooth_spin: QDoubleSpinBox) -> QWidget:
    """Create a simple Qt side panel with instructions and a save button."""
    panel = QWidget()
    layout = QVBoxLayout()
    info = QLabel(
        "Left click: add point\n"
        "Delete/Backspace: remove selected\n"
        "S key or button: extract prompts"
    )
    info.setWordWrap(True)
    layout.addWidget(info)

    form = QFormLayout()
    form.addRow("Display smoothing (σ)", smooth_spin)
    layout.addLayout(form)
    layout.addWidget(invert_checkbox)

    save_btn = QPushButton("Extract selected prompts")
    save_btn.clicked.connect(on_save)
    layout.addWidget(save_btn)
    panel.setLayout(layout)
    return panel


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


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Napari GUI for prompt-based picking.")
    parser.add_argument("--tomo", required=True, type=Path, help="Path to input tomogram (e.g., .mrc).")
    parser.add_argument(
        "--output-dir",
        default=Path("prompt_outputs"),
        type=Path,
        help="Directory to save extracted subtomos and prompt metadata (will be created).",
    )
    parser.add_argument(
        "--invert-contrast",
        action="store_true",
        help="IMPORTANT: ProPicker assumes particles are bright on dark background. If your tomogram has dark particles on bright background, use this flag to invert the contrast.",
    )
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

    volume = load_tomogram(args.tomo)
    base_volume = volume.copy()
    contrast_limits = percentile_contrast(base_volume, 1, 99)

    viewer = napari.Viewer(title=f"Prompt picker - {args.tomo.name}")
    image_layer = viewer.add_image(
        base_volume,
        name=args.tomo.name,
        colormap="gray",
        contrast_limits=contrast_limits,
    )
    smooth_spin = QDoubleSpinBox()
    smooth_spin.setRange(0.0, 10.0)
    smooth_spin.setSingleStep(0.2)
    smooth_spin.setValue(0.0)

    invert_checkbox = QCheckBox("Invert contrast (bright particles)")
    invert_checkbox.setChecked(args.invert_contrast)

    def apply_smoothing(sigma: float) -> None:
        """Update display image with Gaussian smoothing (display only)."""
        img = gaussian_filter(base_volume, sigma) if sigma > 0 else base_volume
        image_layer.data = img
        image_layer.contrast_limits = contrast_limits

    def apply_invert(checked: bool) -> None:
        image_layer.colormap = "gray_r" if checked else "gray"
        viewer.status = (
            "Contrast inverted: particles should be white on dark."
            if checked
            else "Contrast normal. Toggle 'Invert contrast' if particles are dark and should appear white on dark."
        )

    invert_checkbox.toggled.connect(apply_invert)
    smooth_spin.valueChanged.connect(lambda v: apply_smoothing(v))
    apply_smoothing(smooth_spin.value())
    apply_invert(args.invert_contrast)
    
    boxes_layer = viewer.add_shapes(
        [],
        name="prompt_boxes",
        shape_type="rectangle",
        edge_color="yellow",
        face_color="transparent",
        blending="translucent",
        ndim=3,
    )
    boxes_layer.editable = False
    boxes_layer.interactive = False

    points_layer = viewer.add_points(
        np.empty((0, 3)),
        name="prompts",
        size=6,
        face_color="red",
        ndim=3,
    )
    points_layer.mode = "add"
    
    def update_boxes(event=None):
        """Draw a 37x37 square in the XY plane around each prompt (centered on voxel)."""
        half = (37 - 1) / 2  # half-width in pixels
        rectangles = []
        for z, y, x in points_layer.data:
            rectangles.append(
                [
                    [z, y - half, x - half],
                    [z, y - half, x + half],
                    [z, y + half, x + half],
                    [z, y + half, x - half],
                ]
            )
        boxes_layer.data = rectangles

    update_boxes()
    points_layer.events.data.connect(update_boxes)

    def hide_layer_in_list(layer) -> None:
        """Hide a layer row in the Napari layer list while keeping it visible on canvas."""
        try:
            qt_viewer = getattr(viewer.window, "_qt_viewer", None) or getattr(viewer.window, "qt_viewer", None)
            if qt_viewer is None:
                return
            layer_list = getattr(qt_viewer, "layerList", None) or getattr(qt_viewer, "layers", None)
            if layer_list is None:
                return
            idx = viewer.layers.index(layer)
            layer_list.setRowHidden(idx, True)
        except Exception:
            # Best-effort; if napari API changes, we just leave the layer visible in the list.
            return

    hide_layer_in_list(boxes_layer)

    def on_save(event=None):
        if points_layer.data.size == 0:
            msg = "No prompts to save."
            print(msg)
            viewer.status = msg
            return

        invert_factor = -1 if invert_checkbox.isChecked() else 1
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        picks_path = output_dir / "prompt_coords.tsv"
        save_points(points_layer.data, picks_path, "zyx")
        save_subtomos(points_layer.data, volume, output_dir, size=37, invert_factor=invert_factor)

        msg = (
            f"Saved {len(points_layer.data)} prompts to {picks_path}; "
            f"subtomos and coordinates to {output_dir}; "
            f"invert={invert_checkbox.isChecked()})"
        )
        print(msg)
        viewer.status = msg

    viewer.bind_key("s")(on_save)
    panel = build_panel(on_save, invert_checkbox, smooth_spin)
    viewer.window.add_dock_widget(panel, area="right")
    
    viewer.window.resize(1900, 1100)  # tweak as desired
    napari.run()


if __name__ == "__main__":
    main()
