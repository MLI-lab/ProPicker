#!/usr/bin/env python
"""
CLI for ProPicker inference.

Example:
    python run_propicker.py --tomo data/example.mrc --prompt-subtomos prompt_subtomos --output-dir pred_locmaps
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from propicker.inference import get_pred_locmap_dict
from propicker.model import ProPicker
from propicker.paths import PROPICKER_MODEL_FILE, TOMOTWIN_MODEL_FILE
from propicker.utils.mrctools import load_mrc_data

DEFAULT_SUBTOMO_SIZE = 64
DEFAULT_DATALOADER_WORKERS = 0


def load_prompt_subtomos(prompt_dir: Path) -> Dict[str, torch.Tensor]:
    """Load prompt subtomos from .mrc files into a dict keyed by filename stem."""
    prompt_dir = prompt_dir.expanduser().resolve()
    if not prompt_dir.exists():
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")
    subtomos = {}
    for mrc_path in sorted(prompt_dir.glob("*.mrc")):
        data = load_mrc_data(mrc_path).float()
        if data.ndim != 3:
            raise ValueError(f"Prompt subtomo must be 3D, got shape {tuple(data.shape)} for {mrc_path}")
        if data.shape != (37, 37, 37):
            raise ValueError(f"Prompt subtomo must be 37x37x37, got {tuple(data.shape)} for {mrc_path}")
        subtomos[mrc_path.stem] = data
    if not subtomos:
        raise ValueError(f"No .mrc files found in prompt directory: {prompt_dir}")
    return subtomos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ProPicker inference for one or more tomograms.")
    p.add_argument("--tomo", nargs="+", required=True, type=Path, help="Path(s) to input tomograms (.mrc).")
    p.add_argument(
        "--prompt-subtomos",
        type=Path,
        default=Path("prompt_subtomos"),
        help="Directory containing prompt subtomo .mrc files (e.g., prompt_1.mrc).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pred_locmaps"),
        help="Where to write prediction volumes (one .pt per tomo).",
    )
    p.add_argument(
        "--propicker-ckpt",
        type=Path,
        default=Path(PROPICKER_MODEL_FILE),
        help="Path to ProPicker checkpoint (.ckpt).",
    )
    p.add_argument(
        "--tomotwin-ckpt",
        type=Path,
        default=Path(TOMOTWIN_MODEL_FILE),
        help="Path to TomoTwin weights (for computing prompt embeddings).",
    )
    p.add_argument("--device", default=None, help="Torch device (e.g., cuda, cuda:0, or cpu). Default: auto.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for inference subtomos.")
    p.add_argument("--subtomo-overlap", type=int, default=32, help="Overlap between subtomos.")
    p.add_argument(
        "--invert-contrast",
        action="store_true",
        help="Multiply input tomograms by -1 before inference (leave prompts as-is).",
    )
    return p.parse_args()


def choose_device(device_arg: str | None) -> torch.device:
    """Pick user device or fall back to CUDA if available."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path: Path, device: torch.device) -> ProPicker:
    """Load ProPicker from checkpoint to device."""
    model = ProPicker.load_from_checkpoint(str(ckpt_path), map_location=device)  # type: ignore[arg-type]
    model = model.to(device).eval()
    model.freeze()
    return model


def run_inference_for_tomo(
    tomo_path: Path,
    model: ProPicker,
    prompt_subtomos: Dict[str, torch.Tensor],
    tomotwin_ckpt: Path,
    batch_size: int,
    subtomo_overlap: int,
    invert_contrast: bool,
) -> Dict[str, torch.Tensor]:
    """Run ProPicker on a single tomo path and return prediction dict."""
    tomo_path = tomo_path.expanduser().resolve()
    if not tomo_path.exists():
        raise FileNotFoundError(f"Tomo not found: {tomo_path}")
    print(f"\nProcessing {tomo_path} ...")

    tomo = load_mrc_data(tomo_path).float()
    if invert_contrast:
        tomo = -tomo

    pred_locmap_dict = get_pred_locmap_dict(
        model=model,
        tomo=tomo,
        prompt_subtomos_dict=prompt_subtomos,
        tomotwin_model_file=str(tomotwin_ckpt),
        subtomo_size=DEFAULT_SUBTOMO_SIZE,
        subtomo_overlap=subtomo_overlap,
        batch_size=batch_size,
        num_dataloader_workers=DEFAULT_DATALOADER_WORKERS,
    )
    return pred_locmap_dict


def main() -> None:
    args = parse_args()

    device = choose_device(args.device)
    print(f"Using device: {device}")

    prompt_subtomos = load_prompt_subtomos(args.prompt_subtomos)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ProPicker checkpoint: {args.propicker_ckpt}")
    model = load_model(args.propicker_ckpt, device)

    with torch.no_grad():
        for tomo_path in args.tomo:
            pred_locmap_dict = run_inference_for_tomo(
                tomo_path=tomo_path,
                model=model,
                prompt_subtomos=prompt_subtomos,
                tomotwin_ckpt=args.tomotwin_ckpt,
                batch_size=args.batch_size,
                subtomo_overlap=args.subtomo_overlap,
                invert_contrast=args.invert_contrast,
            )

            stem = tomo_path.stem
            out_pt = args.output_dir / f"{stem}_pred_locmaps.pt"
            torch.save({k: v.cpu() for k, v in pred_locmap_dict.items()}, out_pt)
            print(f"Saved prediction dict -> {out_pt}")

if __name__ == "__main__":
    main()
