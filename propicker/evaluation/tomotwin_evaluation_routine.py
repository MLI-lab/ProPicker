"""
These functions are modified from tomotwin.modules.inference.locator
"""
from tomotwin.modules.common.preprocess import label_filename
import numpy as np
import pandas as pd
import os

def _interval_overlap_vec(x1_min, x1_max, x2_min, x2_max):
    """
    tomotwin.modules.inference.locator
    """
    intersect = np.zeros(shape=(len(x1_min)))
    cond_a = x2_min < x1_min
    cond_b = cond_a & (x2_max >= x1_min)
    intersect[cond_b] = np.minimum(x1_max[cond_b], x2_max[cond_b]) - x1_min[cond_b]
    cond_c = ~cond_a & (x1_max >= x2_min)
    intersect[cond_c] = np.minimum(x1_max[cond_c], x2_max[cond_c]) - x2_min[cond_c]

    return intersect

def _bbox_iou_vec_3d(boxesA: np.array, boxesB: np.array) -> np.array:
    """
    tomotwin.modules.inference.locator
    """
    # 0 x
    # 1 y
    # 2 z
    # 3 w
    # 4 h
    # 5 depth

    x1_min = boxesA[:, 0] - boxesA[:, 3] / 2
    x1_max = boxesA[:, 0] + boxesA[:, 3] / 2
    y1_min = boxesA[:, 1] - boxesA[:, 4] / 2
    y1_max = boxesA[:, 1] + boxesA[:, 4] / 2
    z1_min = boxesA[:, 2] - boxesA[:, 5] / 2
    z1_max = boxesA[:, 2] + boxesA[:, 5] / 2

    x2_min = boxesB[:, 0] - boxesB[:, 3] / 2
    x2_max = boxesB[:, 0] + boxesB[:, 3] / 2
    y2_min = boxesB[:, 1] - boxesB[:, 4] / 2
    y2_max = boxesB[:, 1] + boxesB[:, 4] / 2
    z2_min = boxesB[:, 2] - boxesB[:, 5] / 2
    z2_max = boxesB[:, 2] + boxesB[:, 5] / 2

    intersect_w = _interval_overlap_vec(x1_min, x1_max, x2_min, x2_max)
    intersect_h = _interval_overlap_vec(y1_min, y1_max, y2_min, y2_max)
    intersect_depth = _interval_overlap_vec(z1_min, z1_max, z2_min, z2_max)
    intersect = intersect_w * intersect_h * intersect_depth
    union = boxesA[:, 3] * boxesA[:, 4] * boxesA[:, 5] + boxesB[:, 3] * boxesB[:, 4] * boxesB[:,
                                                                                        5] - intersect
    return intersect / union

def locate_positions_stats(locate_results, class_positions, iou_thresh):
    """
    tomotwin.scripts.evaluation
    """
    class_stats = {}
    locate_results_np =  locate_results[["X", "Y", "Z", "width", "height", "depth"]].to_numpy()
    true_positive = 0
    false_negative = 0
    found = np.array([False] * len(locate_results_np))
    for class_pos in class_positions.to_numpy():

        ones = np.ones((len(locate_results_np), 6))
        class_pos_rep = ones * class_pos
        ious = _bbox_iou_vec_3d(class_pos_rep, locate_results_np)
        iou_mask = ious > iou_thresh

        # if np.count_nonzero(iou_mask) >= 2:
        #     import inspect
        #     callerframerecord = inspect.stack()[1]
        #     frame = callerframerecord[0]
        #     info = inspect.getframeinfo(frame)
            #print(f"{np.count_nonzero(iou_mask)} Maxima?? WAIT WHAT? oO")
        if np.any(iou_mask):

            found[np.argmax(ious)] = True

            true_positive = true_positive + 1
        else:
            false_negative = false_negative + 1
    false_positive = np.sum(np.array(found) == False)
    true_positive_rate = true_positive / len(class_positions)

    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
    class_stats["F1"] = float(f1_score)
    class_stats["Recall"] = recall
    class_stats["Precision"] = float(precision)
    class_stats["TruePositiveRate"] = float(true_positive_rate)
    class_stats["TP"] = int(true_positive)
    class_stats["FP"] = int(false_positive)
    class_stats["FN"] = int(false_negative)
    return class_stats

def _filter(df, min_val=None, max_val=None, field=None):
    """xr
    tomotwin.scripts.evaluation
    """
    dfc = df.copy()
    if field == None:
        return dfc

    if min_val != None:
        dfc = dfc[dfc[field] > min_val]

    if max_val != None:
        dfc = dfc[dfc[field] < max_val]
    return dfc


def get_stats(df, positions, iou_thresh=0.6):
    """
    tomotwin.scripts.evaluation
    """
    #label_filename()

    refs = df.attrs['references']
    pc = df['predicted_class'].iloc[0]
    class_name = os.path.splitext(refs[pc])[0]
    try:
        # sometimes, labels are filenames, this one extracts the class name if necessary
        class_name = label_filename(class_name)
    except:
        pass
    pos_classes = np.array([cl.upper() for cl in positions["class"]])
    class_positions = positions[pos_classes == class_name.upper()]
    class_positions = class_positions[["X", "Y", "Z", "width", "height", "depth"]]
    # locate_results.columns = [["X", "Y", "Z", "class"]]

    df = df.rename(columns={"predicted_class_name": "class"})
    df["class"] = class_name
    if "height" in df.columns and "width" in df.columns and "depth" in df.columns:
        pass
    else:
        print(f"Did not find height, width, depth in locate_results, adding size according to tomotwin SIZE_DICT")
        df = _add_size(df, size=37, size_dict=SIZE_DICT)
    stats = locate_positions_stats(locate_results=df, class_positions=class_positions, iou_thresh=iou_thresh)
    return stats

def optim(locate_results, positions, min_size_range=[1, 500], max_size_range=[1, 500], min_size_step=2, max_size_step=2):
    """
    tomotwin.scripts.evaluation
    """
    def find_best(locate_results, field, range, stepsize, type):

        best_stats = get_stats(locate_results, positions)
        #print(best_stats)
        #import sys
        #sys.exit()
        best_f1 = best_stats["F1"]
        best_value = 0
        best_df = locate_results

        for val in np.arange(start=range[0], stop=range[1], step=stepsize):
            if type == "min":
                df = _filter(locate_results, min_val=val, field=field)
            if type == "max":
                df = _filter(locate_results, max_val=val, field=field)
            if len(df) == 0:
                continue
            stats = get_stats(df, positions)
            if stats["F1"] > best_f1:
                best_f1 = stats["F1"]
                best_stats = stats
                best_value = val
                best_df = df.copy()
        return best_stats, best_df, best_value

    # min_size_range = [1, 500]
    # max_size_range = [1, 500]
    # dsize = 2
    # min_similarity_range = [0,1]
    # dsim = self.stepsize_optim_similarity
    locate_results_id = locate_results
    o_dict = {}
    # stats, locate_results_filtered, best_value = find_best(
    #     locate_results=locate_results_id,
    #     field="metric_best",
    #     range=min_similarity_range,
    #     stepsize=dsim,
    #     type="min"
    # )
    # if locate_results_filtered is not None:
    #     o_dict["O_METRIC"] = float(best_value)
    #     locate_results_id = locate_results_filtered

    stats, locate_results_filtered, best_value = find_best(
        locate_results=locate_results_id,
        field="size",
        range=min_size_range,
        stepsize=min_size_step,
        type="min"
    )
    if locate_results_filtered is not None:
        o_dict["O_MIN_SIZE"] = int(best_value)
        locate_results_id = locate_results_filtered

    stats, locate_results_filtered, best_value = find_best(
        locate_results=locate_results_id,
        field="size",
        range=max_size_range,
        stepsize=max_size_step,
        type="max"
    )
    if locate_results_filtered is not None:
        o_dict["O_MAX_SIZE"] = int(best_value)
        locate_results_id = locate_results_filtered

    stats.update(o_dict)

    return stats


# boxsizes for particles contained in the training data
SIZE_DICT = {
    "1AVO": 18,
    "1FZG": 28,
    "1JPM": 18,
    "2HMI": 14,
    "2VYR": 18,
    "3EWF": 18,
    "1E9R": 21,
    "1OAO": 20,
    "2DF7": 33,
    "5XNL": 33,
    "1UL1": 18,
    "2RHS": 19,
    "3MKQ": 33,
    "7EY7": 25,
    "3ULV": 28,
    "1N9G": 19,
    "7BLQ": 28,
    "6WZT": 27,
    "7EGQ":	25,
    "5VKQ":	30,
    "7LSY":	30,
    "7KDV":	29,
    "6LXV":	28,
    "7DD9":	25,
    "7AMV":	25,
    "7NHS":	24,
    "7E8H":	25,
    "7E1Y":	25,

    "2WW2": 20,
    "7VTQ":	28,
    "6YT5":	30,
    "7EGD":	32,
    "7SN7":	32,
    "7WOO":	35,
    "7MEI":	32,
    "7T3U":	30,
    "6Z6O":	35,
    "7BKC":	31,
    "7EEP":	34,

    "7E8S": 35,
    "7QJ0": 30,
    "7NYZ": 35,
    "6VQK": 22,
    "6ZIU": 30,
    "6X02": 26,
    "7E6G": 21,
    "7O01": 35,
    "6X5Z": 30,
    "7WBT": 21,
    "6VGR": 22,
    "4UIC": 23,
    "6Z3A": 28,
    "7KFE": 18,
    "7WI6": 23,
    "7SHK": 17,
    "5TZS": 37,
    "7EGE": 30,
    "7ETM": 21,
    "6SCJ": 30,
    "6TAV": 20,
    "2VZ9": 23,
    "6KLH": 21,
    "1KB9": 20,
    "3PXI": 18,
    "4YCZ": 18,
    "6IGC": 30,

    "6F8L":	18,
    "6JY0":	25,
    "6TA5":	37,
    "6TGC":	28,
    "2DFS":	30,
    "6KSP":	27,
    "7JSN":	24,
    "6KRK":	20,
    "7NIU":	23,
    "5A20":	35,

    "5OOL":	30,
    "6UP6":	33,
    "6I0D":	25,
    "6BQ1":	30,
    "7SFW":	26,
    "3LUE":	37,
    "6JK8":	20,
    "5H0S":	22,
    "6LX3":	17,
    "5LJO":	21,

    "6DUZ":	32,
    "4XK8":	23,
    "6XF8":	29,
    "6M04":	22,
    "6U8Q":	23,
    "6LXK":	24,
    "6CE7":	20,
    "5CSA":	28,
    "7SGM":	25,
    "7B5S":	25,

    "6GYM":	28,
    "6EMK":	27,
    "6W6M":	19,
    "7R04":	35,
    "5O32":	22,
    "6CES":	23,
    "2XNX":	25,
    "6LMT":	17,
    "7BLR":	25,
    "2R9R":	18,

    "6ZQJ": 24,
    "4WRM": 22,
    "7S7K": 23,
    "4V94": 37,
    "4CR2": 33,
    "1QVR": 25,
    "1BXN": 19,
    "3CF3": 25,
    "1U6G": 18,
    "3D2F": 22,
    "2CG9": 18,
    "3H84": 18,
    "3GL1": 13,
    "3QM1": 12,
    "1S3X": 12,
    "5MRC": 37,

    "1FPY": 18,
    "1FO4": 23,
    "1FZ8": 19,
    "1JZ8": 20,
    "4ZIT": 17,
    "5BK4": 25,
    "5BW9": 25,

    "1CU1": 17,
    "1SS8": 22,
    "6AHU": 21,
    "6TPS": 28,
    "6X9Q": 37,
    "6GY6": 31,
    "6NI9": 12,
    "6VZ8": 25,
    "4HHB": 12,
    "7B7U": 20,

    "6Z80": 18,
    "6PWE": 14,
    "6PIF": 20,
    "6O9Z": 21,
    "6ID1": 30,
    "5YH2": 16,
    "4RKM": 16,
    "1G3I": 16,
    "1DGS": 14,
    "1CLQ": 15,

    "7Q21": 20,
    "7KJ2": 25,
    "7K5X": 18,
    "7FGF": 14,
    "7CRQ": 22,
    "6YBS": 25,
    "5JH9": 23,
    "5A8L": 20,
    "3IF8": 15,
    "2B3Y": 14,

    "6VN1": 14,
    "6MRC": 23,
    "6CNJ": 25,
    "5G04": 26,
    "4QF6": 17,
    "1SVM": 18,
    "1O9J": 17,
    "1ASZ": 19,
    "VESICLE": None,
    "FIDUCIAL": 18
}


def _add_size(df, size, size_dict = SIZE_DICT) -> pd.DataFrame:
    """
    tomotwin.scripts.evaluation
    """
    if size_dict is None:
        size = size
        df["width"] = size
        df["height"] = size
        df["depth"] = size
    else:
        df["width"] = 0
        df["height"] = 0
        df["depth"] = 0
        for row_index, row in df.iterrows():
            try:
                s = size_dict[str(row["class"]).upper()]
            except KeyError:
                #print(f"Can't find {str(row['class']).upper()} in size dict. Use default size {size}")
                s = size
            df.at[row_index, "width"] = s
            df.at[row_index, "height"] = s
            df.at[row_index, "depth"] = s

    return df