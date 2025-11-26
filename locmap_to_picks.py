#!/usr/bin/env python
"""
CLI to convert predicted locmaps into picks via clustering.

Examples:
  # direct thresholds
  python locmap_to_picks.py --pred-locmap pred_locmaps/sample_pred_locmaps.pt --prompt prompt_1 \
    --binarization-threshold 0.5 --min-cluster-size-voxels 1000 --max-cluster-size-voxels 50000

  # thresholds from JSON (keys: binarization_threshold, min_size_voxels, max_size_voxels)
  python locmap_to_picks.py --pred-locmap pred_locmaps/sample_pred_locmaps.pt --prompt prompt_1 \
    --thresholds-json thresholds.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from clustering_and_picking import get_cluster_centroids_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster predicted locmaps into picks.")
    thresh_group = p.add_mutually_exclusive_group(required=True)
    p.add_argument(
        "--pred-locmap",
        nargs="+",
        type=Path,
        required=True,
        help="Path(s) to predicted locmap .pt files or directories containing them.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt name corresponding to entry in locmap dictionary. If omitted and only one key exists, it is used.",
    )
    thresh_group.add_argument(
        "--thresholds-json",
        type=Path,
        default=None,
        help="JSON with keys: binarization_threshold, min_size_voxels, max_size_voxels.",
    )
    thresh_group.add_argument(
        "--thresholds",
        nargs=3,
        metavar=("BIN_THRESH", "MIN_SIZE", "MAX_SIZE"),
        type=float,
        help="Provide thresholds directly: binarization_threshold min_size_voxels max_size_voxels.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pred_picks"),
        help="Where to write predicted picks (.tsv) files.",
    )
    return p.parse_args()


def load_thresholds(args: argparse.Namespace) -> Dict[str, float]:
    """Resolve thresholds from JSON or direct triple."""
    if args.thresholds_json:
        path = args.thresholds_json.expanduser().resolve()
        with open(path, "r") as f:
            loaded = json.load(f)
        missing = [k for k in ("binarization_threshold", "min_size_voxels", "max_size_voxels") if k not in loaded]
        if missing:
            raise ValueError(f"Thresholds JSON missing keys: {missing}")
        return {
            "binarization_threshold": loaded["binarization_threshold"],
            "min_size_voxels": loaded["min_size_voxels"],
            "max_size_voxels": loaded["max_size_voxels"],
        }
    if args.thresholds:
        bin_thresh, min_size, max_size = args.thresholds
        return {
            "binarization_threshold": bin_thresh,
            "min_size_voxels": min_size,
            "max_size_voxels": max_size,
        }
    raise ValueError("No thresholds provided.")


def expand_locmap_paths(paths: Iterable[Path]) -> List[Path]:
    locmap_paths: List[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_dir():
            locmap_paths.extend(sorted(path.glob("*.pt")))
        else:
            locmap_paths.append(path)
    if not locmap_paths:
        raise ValueError("No locmap .pt files found.")
    return locmap_paths


def run_clustering_for_tomo(
    pred_locmap_path: Path,
    prompt: str | None,
    binarization_threshold: float,
    min_cluster_size_voxels: float,
    max_cluster_size_voxels: float,
):
    """Run clustering on a single locmap path and return filtered centroids DataFrame."""
    pred_locmap_path = pred_locmap_path.expanduser().resolve()
    if not pred_locmap_path.exists():
        raise FileNotFoundError(f"Predicted locmap not found: {pred_locmap_path}")
    print(f"\nProcessing {pred_locmap_path} ...")
    pred_locmap_dict = torch.load(pred_locmap_path, map_location="cpu")
    keys = list(pred_locmap_dict.keys()) if isinstance(pred_locmap_dict, dict) else None
    if not isinstance(pred_locmap_dict, dict):
        raise ValueError(f"Expected a dict in {pred_locmap_path}, got {type(pred_locmap_dict)}")
    if prompt is None:
        if len(keys) == 1:
            prompt = keys[0]
            print(f"No prompt provided; using only key: {prompt}")
        else:
            raise ValueError(f"Multiple prompts available; specify one with --prompt. Keys: {keys}")
    if prompt not in pred_locmap_dict:
        raise ValueError(f"Prompt '{prompt}' not found in locmap dict keys: {keys}")
    locmap = pred_locmap_dict[prompt]
    if torch.is_tensor(locmap):
        locmap = locmap.cpu()

    binary_mask = locmap > binarization_threshold
    df = get_cluster_centroids_df(binary_mask)
    print(f"Number of clusters before filtering: {len(df)}")
    df = df[(min_cluster_size_voxels <= df["size"]) & (df["size"] <= max_cluster_size_voxels)]
    print(f"Number of clusters after filtering: {len(df)}")
    return df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args)
    locmap_paths = expand_locmap_paths(args.pred_locmap)

    with torch.no_grad():
        for locmap_path in locmap_paths:
            df = run_clustering_for_tomo(
                pred_locmap_path=locmap_path,
                prompt=args.prompt,
                binarization_threshold=thresholds["binarization_threshold"],
                min_cluster_size_voxels=thresholds["min_size_voxels"],
                max_cluster_size_voxels=thresholds["max_size_voxels"],
            )

            stem = locmap_path.stem
            out_tsv = args.output_dir / f"{stem}_picks.tsv"
            df.to_csv(out_tsv, sep="\t", index=False)
            print(f"Wrote picks to {out_tsv}")


if __name__ == "__main__":
    main()
