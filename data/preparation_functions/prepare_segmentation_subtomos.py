#%%
import os
import sys
sys.path.append("..")
sys.path.append("../..")

from matplotlib import pyplot as plt
import pandas as pd
import json
import glob
import tqdm
import math
import scipy
import torch

from data.subtomos import extract_subtomos
from data.utils import save_h5_tensor_dict, save_mrc_data, load_mrc_data


def prepare_segmentation_subtomos(tomo, locmaps,  out_dir, subtomo_size=37, subtomo_extraction_strides=None, save_full_locmaps=False, class_coord_dict=None, tomotwin_model_file=None, setup_tomotwin_reference_embeddings=True, add_background_class_to_locmap=False, skip_existing=True, crop_tomo_fn=None, device="cuda:0"):
    """
    Setup subtomos for segmentation training

    :param tomo: A single tomogram as 3D torch tensor
    :param locmaps: A dict of locmaps as 3D torch tensors. Each key is a particle name and each value a segmentation map (locmap) that has the same shape as tomo. In the locmap, 1 indicates the presence of the particle and 0 its absence.
    :param out_dir: output directory

    :param subtomo_size: size of the subtomos
    :param subtomo_extraction_strides: A list with 3 entires, each entry is the stride for the corresponding dimension. Small strides lead to more subtomos but also more overlap between them.
    :param save_full_locmaps: If true, the locmaps variable is saved as a h5 file in the out_dir


    The following parameters are for generation of TomoTwin prompts and are only relevant if setup_tomotwin_reference_embeddings=True

    :param setup_tomotwin_reference_embeddings: If false, no TomoTwin prompts are generated and saved
    :param class_coord_dict: Dict of pandas dataframes. Dict keys must be the same as the ones in locmaps. The dict values are dataframes with columns, x, y, and z that contain the coordinates of the particles in the tomogram. 
    :param tomotwin_model_file: path to the tomotwin model file

    """  
    out_subtomo_dir = f"{out_dir}/subtomos/model_inputs"
    out_locmap_subtomo_dir = f"{out_dir}/subtomos/locmaps"
    out_prompt_embeds_dir = f"{out_dir}/prompts/embeds"
    out_prtcl_subtomo_dir = f"{out_dir}/prompts/subtomos"

    # skip if all output directories exist and are not empty
    if skip_existing:
        skip = False
        for d in [out_locmap_subtomo_dir, out_prtcl_subtomo_dir, out_prompt_embeds_dir, out_subtomo_dir]:
            if os.path.exists(d):
                if len(os.listdir(d)) > 0:
                    skip = True
        if skip:
            print(f"Run {out_dir} seems to have been processed already. Skipping.")
            return

        # if all([os.path.exists(d) for d in [out_locmap_subtomo_dir, out_prtcl_subtomo_dir, out_prompt_embeds_dir, out_subtomo_dir]]):
        #     print(f"Run {out_dir} seems to have been processed already. Skipping.")
        #     return
    
    for d in [out_locmap_subtomo_dir, out_prompt_embeds_dir, out_subtomo_dir]:
        os.makedirs(d, exist_ok=True)

    if setup_tomotwin_reference_embeddings:
        # extract particle prompts
        prtcls = extract_prtcls(
            tomo=tomo, 
            class_coord_dict=class_coord_dict, 
            box_size=37,
            discard_incomplete_boxes=True
        )
        for cls in prtcls.keys():
            os.makedirs(f"{out_prtcl_subtomo_dir}/{cls}", exist_ok=True)
            for i, prtcl in enumerate(prtcls[cls]):
                save_mrc_data(prtcl, f"{out_prtcl_subtomo_dir}/{cls}/{i}.mrc")
        # embed particle prompts
        for cls in os.listdir(out_prtcl_subtomo_dir):
            gpu = device.split("cuda:")[1]
            os.system(
                f"CUDA_VISIBLE_DEVICES={gpu} tomotwin_embed.py subvolumes \
                    -m {tomotwin_model_file} \
                    -v {out_prtcl_subtomo_dir}/{cls}/*.mrc \
                    -o {out_prompt_embeds_dir}/{cls}"
            )

    if crop_tomo_fn is not None:  
        original_shape = tomo.shape  
        tomo = crop_tomo_fn(tomo)
        locmaps = {k: crop_tomo_fn(v) for k, v in locmaps.items()}
        print("Cropped tomogram and locmaps from shape", original_shape, "to", tomo.shape)

    # save model input subtomos, -1 because tomotwin needs reversed contrast
    subtomos = extract_subtomos(tomo, subtomo_size=subtomo_size, subtomo_extraction_strides=subtomo_extraction_strides)[0]
    for i, subtomo in tqdm.tqdm(enumerate(subtomos), "Saving model input subtomos", total=len(subtomos)):
        save_mrc_data(subtomo, f"{out_subtomo_dir}/{i}.mrc")
    # background locmap is 1 everywhere where no other locmap is 1
    if add_background_class_to_locmap:
        locmaps = add_background_class(locmaps)
    # save full locmaps
    if save_full_locmaps:
        save_h5_tensor_dict(locmaps, f"{out_dir}/full_locmaps.h5")
    # save locmap subtomos
    for cls in locmaps.keys():
        locmaps[cls] = extract_subtomos(locmaps[cls], subtomo_size=subtomo_size, subtomo_extraction_strides=subtomo_extraction_strides)[0]
    n_locmap_subtomos = len(locmaps[list(locmaps.keys())[0]])
    for k in tqdm.tqdm(range(n_locmap_subtomos), "Saving locmap subtomos"):
        locmap_subtomo_dict = {
            cls: locmaps[cls][k] for cls in locmaps.keys()
        }
        save_h5_tensor_dict(locmap_subtomo_dict, f"{out_locmap_subtomo_dir}/{k}.h5")
        
    


# THINGS BELOW THIS ARE HELPER FUNCTIONS
def extract_subtomo(tomo, coords, size):
    z, y, x = coords
    z = int(z)
    y = int(y)
    x = int(x)
    min_z = max(0, z-size//2)
    max_z = min(tomo.shape[0], z+math.ceil(size/2))
    min_y = max(0, y-size//2)
    max_y = min(tomo.shape[1], y+math.ceil(size/2))
    min_x = max(0, x-size//2)
    max_x = min(tomo.shape[2], x+math.ceil(size/2))
    return tomo[min_z:max_z, min_y:max_y, min_x:max_x]


def extract_prtcls(tomo, class_coord_dict, box_size=37, discard_incomplete_boxes=True):
    prtcls = {
        cls: [] for cls in class_coord_dict.keys()
    }
    incomplete = 0
    for cls, coord in class_coord_dict.items():
        centres = torch.tensor([coord.z.values, coord.y.values, coord.x.values]).T
        for centre in centres:
            subtomo = extract_subtomo(tomo, centre, box_size)
            if discard_incomplete_boxes:
                if subtomo.shape[0] != box_size or subtomo.shape[1] != box_size or subtomo.shape[2] != box_size:
                    incomplete += 1
                    continue
            prtcls[cls].append(subtomo)
    if discard_incomplete_boxes:
        print(f"Discarded {incomplete} incomplete boxes")
    return prtcls

def add_background_class(locmap_dict):
    background_locmap = torch.ones_like(list(locmap_dict.values())[0])
    for cls in locmap_dict.keys():
        background_locmap -= locmap_dict[cls]
        background_locmap = (background_locmap == 1).float()
    locmap_items = list(locmap_dict.items())
    locmap_items = sorted(locmap_items, key=lambda x: x[0], reverse=True)
    locmap_items.insert(0, ("background", background_locmap))
    locmap_dict = dict(locmap_items)
    return locmap_dict
