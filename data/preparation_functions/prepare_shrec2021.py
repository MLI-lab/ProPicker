from .prepare_segmentation_subtomos import prepare_segmentation_subtomos
from data.utils import load_mrc_data
import pandas as pd


def prepare_shrec2021_model(dataset_base_dir, item, out_dir, **kwargs):
    """
    Wrapper for setup_segmentation_subtomos that takes a SHREC2021 model directory as input
    Example for shrec_model_dir: .../full_dataset/model_0
    """
    shrec_model_dir = f"{dataset_base_dir}/{item}"
    locmaps = get_shrec2021_prtcl_locmaps(f"{shrec_model_dir}/class_mask.mrc")
    tomo = -1 * load_mrc_data(f"{shrec_model_dir}/reconstruction.mrc")[170:350] # -1 because tomotwin needs reversed contrast, so I adapted this
    class_coord_dict = get_shrec2021_class_coord_dict(f"{shrec_model_dir}/particle_locations.txt")  # -170 is done in get_shrec2021_class_coord_dict
    prepare_segmentation_subtomos(
        tomo=tomo,
        locmaps=locmaps,
        class_coord_dict=class_coord_dict,
        out_dir=out_dir,
        **kwargs
    )


def get_shrec2021_prtcl_locmaps(shrec_occupancy_class_file):
    id_to_class = {
        #0: "background",
        1: "4v94",  # somehow not in the dataset
        2: "4cr2", 
        3: "1qvr", 
        4: "1bxn", 
        5: "3cf3", 
        6: "1u6g", 
        7: "3d2f", 
        8: "2cg9", 
        9: "3h84", 
        10: "3gl1", 
        11: "3qm1", 
        12: "1s3x", 
        13: "5mrc", 
    }
    occ = load_mrc_data(shrec_occupancy_class_file)[170:350]
    locmaps = {
        id_to_class[i]: (occ == i).float() for i in id_to_class.keys()
    }
    return locmaps

def get_shrec2021_class_coord_dict(coord_file):
    coord = pd.read_csv(coord_file, sep=" ", header=None)
    coord.columns = ["class", "x", "y", "z", "alpha", "beta", "gamma"]
    coord.z -= 170
    # revmove columsn with cls = vesicle or fiducial
    coord = coord[~coord["class"].isin(["vesicle", "fiducial"])]
    class_coord_dict = {
        cls: coord[coord["class"] == cls] for cls in coord["class"].unique()
    }
    return class_coord_dict