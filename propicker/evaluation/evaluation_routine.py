import sys
sys.path.append("..")

import torch
import numpy as np
import pandas as pd

import os
import tqdm
import json
import torch
from .tomotwin_evaluation_routine import SIZE_DICT, get_stats
from propicker.clustering_and_picking.clustering import get_cluster_centroids_df
import multiprocessing
from multiprocessing import Pool

def get_best_case_cluster_based_picking_performance(pred_locmap_dict, gt_positions, n_size_steps, n_thresh_steps=None, iou_thresh=0.6, metric="F1", optimize_thresh=True, outfile=None, min_thresh=None, max_thresh=None, skip_classs=["background"], num_workers=None):
    if not "width" in gt_positions.columns:
        raise ValueError("'width' not specified in gt_positions, but is needed for IoU calculation")
    if not "height" in gt_positions.columns:
        raise ValueError("'height' not specified in gt_positions, but is needed for IoU calculation")
    if not "depth" in gt_positions.columns:
        raise ValueError("'depth' not specified in gt_positions, but is needed for IoU calculation")
    
    jobs, clss = [], []
    pbar_position = 0
    for cls in pred_locmap_dict.keys():
        if cls in skip_classs:
            continue
        job = {}
        pred_locmap = pred_locmap_dict[cls]
        job = {
            "pred_locmap": pred_locmap,
            "gt_positions": gt_positions,
            "cls": cls,
            "optimize_thresh": optimize_thresh,
            "n_thresh_steps": n_thresh_steps,
            "n_size_steps": n_size_steps,
            "iou_thresh": iou_thresh,
            "min_thresh": min_thresh,
            "max_thresh": max_thresh,
            "metric": metric,
            "pbar_position": pbar_position
        }
        jobs.append(job)
        clss.append(cls)
        pbar_position += 1

    num_workers = len(jobs) if num_workers is None else num_workers

    if num_workers == 1 or num_workers == 0:
        print(f"Running evaluation with 1 workers")
        best_stats = []
        for job in jobs:
            best_stats.append(_get_best_case_cluster_based_picking_performance_wrapper(job))
    else:
        #eval = lambda x: _get_best_case_cluster_based_picking_performance(**x)
        with Pool(num_workers) as p:
            best_stats = p.map(_get_best_case_cluster_based_picking_performance_wrapper, jobs)
    best_stats = dict(zip(clss, best_stats))       

    # save cluster based stats to json
    if outfile is not None:
        with open(outfile, "w") as f:
            json.dump(best_stats, f, indent=4)
    return best_stats

def _get_best_case_cluster_based_picking_performance_wrapper(args):
    return _get_best_case_cluster_based_picking_performance(**args)

def _get_best_case_cluster_based_picking_performance(pred_locmap, gt_positions, cls, optimize_thresh, n_thresh_steps, n_size_steps, iou_thresh, min_thresh, max_thresh, metric, pbar_position=0):
    """
    Evaluate the performance of a predicted locmap using clustering and picking.
    """
    if optimize_thresh:
        min_thresh = pred_locmap.min() if min_thresh is None else min_thresh
        max_thresh = pred_locmap.max() if max_thresh is None else max_thresh
        threshs = np.linspace(min_thresh, max_thresh, n_thresh_steps+1, endpoint=False)[1:]
        print(f"Optimizing threshold for {cls} between {min_thresh} and {max_thresh}")
    else:
        threshs = [1.0]

    best_stats = {metric: 0.0}
    pbar = tqdm.tqdm(total=len(threshs) * n_size_steps**2, desc=f"{cls} (Best {metric}: ?.??)", position=pbar_position, leave=True)
    for thresh in threshs:
        pred_locmap_thresh = (pred_locmap >= thresh).float().numpy()
        if pred_locmap_thresh.sum() == 0:
            continue
        df_located = get_cluster_centroids_df(
            binary_locmap=pred_locmap_thresh,
            min_cluster_size=0,
            max_cluster_size=np.inf,
            connectivity=1
    )
        best_stats_ = optimize_cluster_sizes(
            df_located,
            gt_positions, 
            cls=cls, 
            iou_thresh=iou_thresh, 
            n_size_steps=n_size_steps, 
            metric=metric, 
            pbar=pbar,
            update_pbar_desc=False
        )
        if best_stats_[metric] > best_stats[metric]:
            best_stats = best_stats_
            best_stats["thresh"] = float(thresh)
            pbar.set_description(f"{cls} (Best {metric}: {best_stats[metric]:.2f})")
    return best_stats


def optimize_cluster_sizes(df_located, gt_positions, cls, iou_thresh=0.6, n_size_steps=10, metric="F1", pbar=None, update_pbar_desc=True):
    if "x" in gt_positions.columns and "y" in gt_positions.columns and "z" in gt_positions.columns and not "X" in gt_positions.columns and not "Y" in gt_positions.columns and not "Z" in gt_positions.columns:
        gt_positions = gt_positions.rename(columns={"x": "X", "y": "Y", "z": "Z"})
    if "x" in df_located.columns and "y" in df_located.columns and "z" in df_located.columns and not "X" in df_located.columns and not "Y" in df_located.columns and not "Z" in df_located.columns:
        df_located = df_located.rename(columns={"x": "X", "y": "Y", "z": "Z"})
    
    df_located["metric_best"] = 1.0
    df_located["predicted_class"] = 0  # important, only 1 class in this case
    row = gt_positions[gt_positions["class"] == cls].iloc[0]
    df_located["height"] = int(row["height"])
    df_located["width"] = int(row["width"])
    df_located["depth"] = int(row["depth"])
    df_located.attrs["references"] = [cls]

    cluster_size_list = df_located["size"].values
    min_size_range = np.linspace(min(min(cluster_size_list),1), np.quantile(cluster_size_list, 0.45), n_size_steps)
    max_size_range = np.linspace(np.quantile(cluster_size_list, 0.55), np.max(cluster_size_list), n_size_steps)
    
    best_stats = {metric: 0.0}
    for min_size in min_size_range:
        for max_size in max_size_range:
            df_located_ = df_located[df_located["size"].between(min_size, max_size)]
            if min_size < max_size and len(df_located_) > 0:
                stats = get_stats(df_located_, gt_positions, iou_thresh=iou_thresh)
                if stats[metric] > best_stats[metric]:
                    best_stats = stats
                    best_stats["min_size"] = float(min_size)
                    best_stats["max_size"] = float(max_size)
                    best_stats["positions"] = df_located_.to_string()
                    if pbar is not None and update_pbar_desc:
                        pbar.set_description(f"{cls} (Best {metric}: {best_stats[metric]:.2f})")
            if pbar is not None:
                pbar.update(1)
    return best_stats




def evaluate_picks(pred_positions, gt_positions, iou_thresh=0.6):
    if not "width" in gt_positions.columns:
        raise ValueError("'width' not specified in gt_positions, but is needed for IoU calculation")
    if not "height" in gt_positions.columns:
        raise ValueError("'height' not specified in gt_positions, but is needed for IoU calculation")
    if not "depth" in gt_positions.columns:
        raise ValueError("'depth' not specified in gt_positions, but is needed for IoU calculation")
    # these are dummy values needed for interpoarability with the TomoTwin evaluation routine
    pred_positions["metric_best"] = 1.0
    pred_positions["predicted_class"] = 0  # important, only 1 class in this case

    # get all classes in pred_positions
    clss = pred_positions["class"].unique()
    
    results = {}
    for cls in clss:
        pred_positions_class = pred_positions[pred_positions["class"] == cls]
        gt_positions_class = gt_positions[gt_positions["class"] == cls]
        # add size to pred_positions
        row = gt_positions_class.iloc[0]
        pred_positions_class["height"] = int(row["height"])
        pred_positions_class["width"] = int(row["width"])
        pred_positions_class["depth"] = int(row["depth"])
        pred_positions_class.attrs["references"] = [cls]  # need this for TomoTwin evaluation routine
        stats = get_stats(pred_positions_class, gt_positions_class, iou_thresh=iou_thresh)
        results[cls] = stats
        
    return results
