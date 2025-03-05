import sys
sys.path.append("..")
import os
import shutil
from utils.mrctools import load_mrc_data, save_mrc_data
import glob
from data.preparation_functions.prepare_empiar10988 import empiar10988_ts_to_slice_of_interest, get_empiar10988_coord_dict
from paths import EMPIAR10988_BASE_DIR, PROPICKER_MODEL_FILE

# disable warnings for chained assignments in pandas
import pandas as pd
pd.options.mode.chained_assignment = None  

# which TS to use for training and validation
train_ts = ["TS_029"]
val_ts = ["TS_030"]
# crop train_ts to cube 2*crop_delta x 2*crop_delta x 2*crop_delta
crop_delta = 64 
# number of epochs to train on for each crop size
crop_delta_num_epochs_dict = {
    64: 75,
    128: 50,
    256: 25,
    512: 15,
}
# if true, use the binary labels contained in the empiar10988 dataset; we used those for the experiments in the paper
use_binary_labels = True
# gpu for training
gpu_id = 2
# output directory
out_dir = f"./fine_tuning_empiar10988/crop_delta={crop_delta}"

# create output directory
if os.path.exists(out_dir):
    if out_dir.endswith("/trash"):
        shutil.rmtree(out_dir)
    else:
        raise AssertionError(f"Output directory {out_dir} already exists. Please delete it first.")
os.makedirs(out_dir)


# load tomograms, coordinates (and binary labels) and crop them
ts_id_gt_locmap_dict = {}
for id, ts_id in enumerate(train_ts + val_ts):
    # load tomogram
    tomo_file = f"{EMPIAR10988_BASE_DIR}/tomograms/{ts_id}.rec"
    slice_of_interest = empiar10988_ts_to_slice_of_interest[ts_id]
    tomo = -1 * load_mrc_data(tomo_file).float()[slice_of_interest]  # IMPORTANT: tomogram has to be scaled to contrast 'white particles on dark background'!
    
    # setup cropping function for tomogram (and binary labels)
    z_slice = empiar10988_ts_to_slice_of_interest[ts_id]   
    z_width = z_slice.stop - z_slice.start
    z_center = z_width // 2
    z_limit = [z_center - crop_delta, z_center + crop_delta]
    y_limit = [464-crop_delta, 464+crop_delta]
    x_limit = [480-crop_delta, 480+crop_delta]
    x_limit = [max(0, x_limit[0]), min(x_limit[1], tomo.shape[2])]
    y_limit = [max(0, y_limit[0]), min(y_limit[1], tomo.shape[1])]
    z_limit = [max(0, z_limit[0]), min(z_limit[1], tomo.shape[0])]
    def crop_tomo_fn(tomo):
        return tomo[
            z_limit[0]:z_limit[1],
            y_limit[0]:y_limit[1],
            x_limit[0]:x_limit[1]
        ]
    
    # crop tomogram
    tomo = crop_tomo_fn(tomo)
    
    # save cropped tomogram
    if not os.path.exists(f"{out_dir}/raw_data"):
        os.makedirs(f"{out_dir}/raw_data")
    save_mrc_data(tomo, f"{out_dir}/raw_data/{ts_id}.mrc")

    # load particle coordinates
    coords_dir = f"{EMPIAR10988_BASE_DIR}/particle_lists"
    coord_dict = get_empiar10988_coord_dict(coords_dir, ts_id)
    coord_dict = {"cyto_ribosome": coord_dict["cyto_ribosome"]}
    for k, coords in coord_dict.items():
        coords = coords[(coords.x >= x_limit[0]) & (coords.x < x_limit[1]) & (coords.y >= y_limit[0]) & (coords.y < y_limit[1]) & (coords.z >= z_limit[0]) & (coords.z < z_limit[1])]
        coords.x -= x_limit[0]
        coords.y -= y_limit[0]
        coords.z -= z_limit[0]
        coord_dict[k] = coords
        
    coords = coords[["x", "y", "z"]]
    coords.to_csv(f"{out_dir}/raw_data/{ts_id}.coords", index=False, sep="\t", header=None)

    # load binary labels
    if use_binary_labels:
        locmap_file = f"{EMPIAR10988_BASE_DIR}/labels/{ts_id}_cyto_ribosomes.mrc"
        gt_locmap_dict = {"cyto_ribosome": load_mrc_data(locmap_file)[slice_of_interest].float()}
        gt_locmap_dict = {k: crop_tomo_fn(v) for k, v in gt_locmap_dict.items()}
        ts_id_gt_locmap_dict[ts_id] = gt_locmap_dict["cyto_ribosome"]
            

# generate DeepETPicker preprocessing config
os.makedirs(f"{out_dir}/configs", exist_ok=True)
pre_config_file = f"{out_dir}/configs/preprocess.py"
lines = [
    "pre_config={",
    f"\"dset_name\": \"empiar10988_preprocess\",",
    f"\"base_path\": \"{out_dir}\",",
    f"\"coord_path\": \"{out_dir}/raw_data\",",
    f"\"coord_format\": \".coords\",",
    f"\"tomo_path\": \"{out_dir}/raw_data\",",
    f"\"tomo_format\": \".mrc\",",
    f"\"num_cls\": 1,",
    f"\"label_type\": \"gaussian\",",
    f"\"label_diameter\": 21,",
    f"\"ocp_type\": \"sphere\",",
    f"\"ocp_diameter\": \"21\",",
    f"\"norm_type\": \"standardization\"",
    "}"
]
for line in lines:
    with open(pre_config_file, "a") as f:
        f.write(line + "\n")
os.system(f"python ./DeepETPicker_ProPicker/bin/preprocess.py --pre_configs {pre_config_file}")

# generate DeepETPicker training config
os.system(
    f"python ./DeepETPicker_ProPicker/bin/generate_train_config.py \
        --pre_configs '{pre_config_file}' \
        --dset_name 'train' \
        --cfg_save_path '{out_dir}/configs' \
        --train_set_ids {'0' if len(train_ts)==1 else f'0-{len(train_ts)-1}'} \
        --val_set_ids {f'{len(train_ts)}' if len(val_ts) == 1 else f'{len(train_ts)}-{len(train_ts)+len(val_ts)-1}'} \
        --batch_size 8 \
        --block_size 72 \
        --pad_size 12 \
        --learning_rate 1e-3 \
        --max_epoch {crop_delta_num_epochs_dict[crop_delta]} \
        --gpu_id {gpu_id} \
    "
)

# if use_binary_labels, replace DeepETPicker's gaussian labels with binary labels
if use_binary_labels:    
    label_dir = glob.glob(f"{out_dir}/gaussian*")
    if len(label_dir) != 1:
        raise AssertionError(f"Expected to find exactly one label directory, but found {len(label_dir)}")

    for ts_id, locmap in ts_id_gt_locmap_dict.items():
        save_mrc_data(locmap, f"{label_dir[0]}/{ts_id}.mrc")

# run training
os.system(
    f"python ./DeepETPicker_ProPicker/bin/train_bash.py \
        --train_configs '{out_dir}/configs/train.py' \
        --network ProPicker \
        --propicker_model_file '{PROPICKER_MODEL_FILE}' \
        --prompt_embed_file './fixed_prompts_empiar10988.json' \
        --loss_func_seg 'CE' \
    "
)

# remove raw data
shutil.rmtree(f"{out_dir}/raw_data")