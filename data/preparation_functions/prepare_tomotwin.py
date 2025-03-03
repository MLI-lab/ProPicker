import pandas as pd
import json
import glob
import scipy
import torch

from data.utils import center_crop_tomo_to_shape, load_mrc_data
from .prepare_segmentation_subtomos import prepare_segmentation_subtomos


def prepare_tomotwin_run(dataset_base_dir, item, out_dir, **kwargs):
    """
    Wrapper for setup_segmentation_subtomos that takes a TomoTwin run directory as input
    Exampe for run_dir: '.../tomotwin_data/tomograms/tomo_simulation_round_1/tomo_01.2022-04-11T140327+0200'
    """
    run_dir = f"{dataset_base_dir}/{item}"
    locmaps = get_tomotwin_prtcl_locmaps(run_dir)
    tomo = -1 * load_mrc_data(f"{run_dir}/tiltseries_rec.mrc")  # -1 because tomotwin needs reversed contrast, so I adapted this
    class_coord_dict = get_tomotwin_class_coord_dict(f"{run_dir}/coords")    
    # due to voxel size mismatch, locmaps are zooomed in and cropped, therefore we also have 
    off_z, off_y, off_x = ((torch.tensor(tomo.shape) - torch.tensor(locmaps[list(locmaps.keys())[0]].shape))/2).tolist()
    for cls, coords in class_coord_dict.items():
        class_coord_dict[cls]["z"] = coords["z"] - off_z
        class_coord_dict[cls]["y"] = coords["y"] - off_y
        class_coord_dict[cls]["x"] = coords["x"] - off_x
    # now crop the tomogram 
    tomo = center_crop_tomo_to_shape(tomo, locmaps[list(locmaps.keys())[0]].shape)
    prepare_segmentation_subtomos(
        tomo=tomo,
        locmaps=locmaps,
        class_coord_dict=class_coord_dict,
        out_dir=out_dir,
        **kwargs
    )
    


def load_tomotwin_coord_file(file):
    coords = pd.read_csv(
    file, 
    sep=" ",
    header = None,
    skiprows=3
    )
    # no typo, y and x are switched in the file
    coords.columns = ["y", "x", "z", "alpha", "beta", "gamma"]

    coords["x"] *= 10/10.2
    coords["y"] *= 10/10.2
    coords["z"] *= 10/10.2

    # convert coordinate origin from center to corner
    coords["x"] = coords["x"] + 512//2
    coords["y"] = coords["y"] + 512//2
    coords["z"] = coords["z"] + 200//2
    # reverse x coordinate
    coords["x"] = 512 - coords["x"]
    return coords

def get_tomotwin_class_coord_dict(coords_dir):
    coord_files = glob.glob(f"{coords_dir}/*_coords.txt")
    coord_files = [
        f for f in coord_files if not f.endswith("vesicle_coords.txt") and not f.endswith("fiducial_coords.txt")
    ]
    classes = [coord_file.split("/")[-1].split("_coords.txt")[0] for coord_file in coord_files]
    class_coord_dict = {cls: load_tomotwin_coord_file(coord_file) for coord_file, cls in zip(coord_files, classes)}
    return class_coord_dict

def get_tomotwin_prtcl_locmaps(run_dir, zoom_factor=10/10.2, exclude_vesicles=True, exclude_fiducials=True):
    """
    Occu maps have voxel size 10A but tomo has voxel size 10.2A, so we need to zoom the occu maps
    """
    occu = load_mrc_data(f"{run_dir}/coords/occupancy.mrc")

    occu_map_file = f"{run_dir}/coords/occu_map.json"
    with open(occu_map_file, "r") as f:
        occu_map = json.load(f)

    occu_map = {int(k): v for k, v in occu_map.items()}

    exclude = []
    if exclude_vesicles:
        exclude.append("vesicle")
    if exclude_fiducials:
        exclude.append("fiducial")
        
    locmaps = {}
    for id, cls in occu_map.items():
        if cls in exclude:
            continue
        else:   
            locmap = (occu == id).float()
            locmap = scipy.ndimage.zoom(locmap, zoom_factor, order=0, mode="nearest") 
            locmap = torch.tensor(locmap > 0).float()
            locmaps[cls] = locmap
    return locmaps
