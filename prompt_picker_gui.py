#!/usr/bin/env python
"""
Simple Napari GUI to pick prompt locations in a tomogram and extract subtomos around them.

Usage example:
    python prompt_picker_gui.py --tomo path/to/volume.mrc --output picks.tsv --export-order xyz --subtomo-dir prompt_subtomos [--invert-contrast]

Controls:
  - Scroll mouse wheel / use Z slider to move through slices.
  - Left click to add a point (points are stored in Napari's Z, Y, X order).
  - Select a point and press Delete/Backspace to remove.
  - Press "Extract selected prompts" in the side panel or hit the "S" key to write to disk.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable
import sys

# Prefer PySide6 and a headless-friendly Qt platform by default.
os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")

try:
    import mrcfile
    import napari
    import numpy as np
    from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget, QCheckBox
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


def axis_order(value: str) -> str:
    """Validate an axis order string (permutation of zyx)."""
    value = value.lower()
    if sorted(value) != ["x", "y", "z"] or len(value) != 3:
        raise argparse.ArgumentTypeError("Axis order must be a permutation of z, y, x (e.g., zyx or xyz).")
    return value


def load_tomogram(path: Path) -> np.ndarray:
    """Load a tomogram as a float32 3D array."""
    if not path.exists():
        raise FileNotFoundError(f"Tomogram not found: {path}")
    with mrcfile.open(path, permissive=True) as fh:
        data = np.asarray(fh.data)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape}")
    return data.astype(np.float32, copy=False)


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
    start = center - half
    stop = center + half + 1  # exclusive

    subtomo = np.zeros((size, size, size), dtype=volume.dtype)

    src_slices = []
    dst_slices = []
    for axis in range(3):
        src_start = max(start[axis], 0)
        src_stop = min(stop[axis], volume.shape[axis])
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


def build_panel(on_save, invert_checkbox: QCheckBox) -> QWidget:
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
    layout.addWidget(invert_checkbox)
    save_btn = QPushButton("Extract selected prompts")
    save_btn.clicked.connect(on_save)
    layout.addWidget(save_btn)
    panel.setLayout(layout)
    return panel


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Napari GUI for prompt-based picking.")
    parser.add_argument("--tomo", required=True, type=Path, help="Path to input tomogram (e.g., .mrc).")
    parser.add_argument("--output", required=True, type=Path, help="Where to save picked points (TSV).")
    parser.add_argument(
        "--export-order",
        default="zyx",
        type=axis_order,
        help="Axis order for saved points. Napari stores points as Z,Y,X; use xyz if your downstream expects X,Y,Z.",
    )
    parser.add_argument("--name", default=None, help="Optional layer name for the tomogram.")
    parser.add_argument(
        "--subtomo-dir",
        default=Path("prompt_subtomos"),
        type=Path,
        help="Directory to save extracted subtomos (will be created).",
    )
    parser.add_argument(
        "--invert-contrast",
        action="store_true",
        help="Invert display contrast so particles appear bright on dark background.",
    )
    args = parser.parse_args(argv)

    volume = load_tomogram(args.tomo)

    viewer = napari.Viewer(title=f"Prompt picker - {args.tomo.name}")
    image_layer = viewer.add_image(volume, name=args.name or args.tomo.name, colormap="gray", contrast_limits=None)

    invert_checkbox = QCheckBox("Invert contrast (bright particles)")
    invert_checkbox.setChecked(args.invert_contrast)

    def apply_invert(checked: bool) -> None:
        image_layer.colormap = "gray_r" if checked else "gray"
        viewer.status = (
            "Contrast inverted: particles should be white on dark."
            if checked
            else "Contrast normal. Toggle 'Invert contrast' if particles are dark and should appear white on dark."
        )

    invert_checkbox.toggled.connect(apply_invert)
    apply_invert(args.invert_contrast)

    points_layer = viewer.add_points(
        np.empty((0, 3)),
        name="prompts",
        size=6,
        face_color="red",
        ndim=3,
    )
    points_layer.mode = "add"

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
    try:
        qt_layers = getattr(viewer.window.qt_viewer, "layerList", None) or getattr(viewer.window.qt_viewer, "layers", None)
        if qt_layers is not None:
            idx = viewer.layers.index(boxes_layer)
            qt_layers.setRowHidden(idx, True)
    except Exception:
        # If hiding fails on this napari version, ignore; boxes remain visible in the list.
        pass

    def on_save(event=None):
        if points_layer.data.size == 0:
            msg = "No prompts to save."
            print(msg)
            viewer.status = msg
            return

        invert_factor = -1 if invert_checkbox.isChecked() else 1
        save_points(points_layer.data, args.output, args.export_order)
        save_subtomos(points_layer.data, volume, args.subtomo_dir, size=37, invert_factor=invert_factor)

        msg = (
            f"Saved {len(points_layer.data)} prompts to {args.output} "
            f"and subtomos to {args.subtomo_dir} (order: {args.export_order.upper()}; "
            f"invert={invert_checkbox.isChecked()})"
        )
        print(msg)
        viewer.status = msg

    viewer.bind_key("s")(on_save)
    panel = build_panel(on_save, invert_checkbox)
    viewer.window.add_dock_widget(panel, area="right")
    
    viewer.window.resize(1900, 1100)  # tweak as desired
    napari.run()


if __name__ == "__main__":
    main()
