#%%
import pandas as pd
import json
import glob
import scipy
import torch

from data.utils import center_crop_tomo_to_shape, load_mrc_data
from .prepare_segmentation_subtomos import prepare_segmentation_subtomos

#%%
def prepare_empiar10988_ts(dataset_base_dir, item, out_dir, label_subdir="labels", **kwargs):
    """
    example: empiar10988_ts_id = "TS_30"
    empiar10988_base_dir = ".../empiar/10988/DEF"
    """
    slice_of_interest = empiar10988_ts_to_slice_of_interest[item]
    locmap_file = f"{dataset_base_dir}/{label_subdir}/{item}_cyto_ribosomes.mrc"
    locmaps = {"cyto_ribosome": load_mrc_data(locmap_file)[slice_of_interest].float()}
    # load tomogram
    tomo_file = f"{dataset_base_dir}/tomograms/{item}.rec"
    tomo = -1 * load_mrc_data(tomo_file).float()[slice_of_interest]  # -1 because tomotwin needs reversed contrast, so I adapted this
    # load coordinates
    coords_dir = f"{dataset_base_dir}/particle_lists"
    class_coord_dict = get_empiar10988_coord_dict(coords_dir, item)
    prepare_segmentation_subtomos(
        tomo=tomo,
        locmaps=locmaps,
        class_coord_dict=class_coord_dict,
        out_dir=out_dir,
        **kwargs,
    )
    
    
empiar10988_ts_to_slice_of_interest = {
    "TS_026": slice(406, 573),
    "TS_027": slice(130, 330),
    "TS_028": slice(91, 406),
    "TS_029": slice(110, 360),
    "TS_030": slice(120, 347),
    "TS_034": slice(124, 392),
    "TS_037": slice(154, 385),
    "TS_041": slice(136, 378),
    "TS_043": slice(181, 334),
    "TS_045": slice(98, 402),
}

def get_empiar10988_coord_dict(coords_dir, empiar10988_ts_id):
    coord_dict = {}
    coord_files = glob.glob(f"{coords_dir}/{empiar10988_ts_id}*.csv")
    for coord_file in coord_files:
        coords = pd.read_csv(coord_file, sep=",", header=None)
        coords = coords.astype(int)
        coords.columns = ["x", "y", "z"]
        if coord_file.endswith("cyto_ribosomes.csv"):
            clss = "cyto_ribosome"
        elif coord_file.endswith("fas.csv"):
            clss = "fas"
        if clss == "fas":
            continue
        coords["class"] = clss
        coords["rx"] = "NaN"
        coords["ry"] = "NaN"
        coords["rz"] = "NaN"
        coords = coords[["class", "x", "y", "z", "rx", "ry", "rz"]]
        coords["z"] -= empiar10988_ts_to_slice_of_interest[empiar10988_ts_id].start
        coord_dict[clss] = coords
    return coord_dict


def read_empiar10988_coords(coord_file):
    coords = pd.read_csv(coord_file, sep=",", header=None)
    coords = coords.astype(int)
    coords.columns = ["X", "Y", "Z"]
    coords["class"] = "cyto_ribosome"
    coords["rx"] = "NaN"
    coords["ry"] = "NaN"
    coords["rz"] = "NaN"
    coords = coords[["class", "X", "Y", "Z", "rx", "ry", "rz"]]
    return coords
