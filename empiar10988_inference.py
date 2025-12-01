#%%
import os
import shutil
import importlib.util
import copy

from propicker.utils.mrctools import load_mrc_data, save_mrc_data
from propicker.data.preparation_functions.prepare_empiar10988 import empiar10988_ts_to_slice_of_interest, get_empiar10988_coord_dict
from propicker.paths import EMPIAR10988_BASE_DIR, PROPICKER_MODEL_FILE

# TS for testing, you can add more TS if you want
test_ts = ["TS_030"]
# modify this to point to the checkpoint file you want to test
ckpt_file = "./fine_tuning_empiar10988/crop_delta=64/runs/train/train_ProPicker_BlockSize72_CELoss_MaxEpoch100_bs8_lr0.001_IP1_bg1_coord1_Softmax0_bn__TNNone/version_0/checkpoints/epoch=29-step=269.ckpt"
# modify this to point to the train config file you used for training
train_cfg_file = "./fine_tuning_empiar10988/crop_delta=64/configs/train.py"

gpu = 0
batch_size = 16

# data will be temporarily saved in this directory; it will be removed after testing
tmp_dir = f"./fine_tuning_empiar10988/crop_delta=64/test"
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(f"{tmp_dir}/raw_data")

# load and sav test data
ts_id_gt_locmap_dict = {}
ts_id_coord_dict = {}
for ts_id in test_ts:
    # load_tomogram
    tomo_file = f"{EMPIAR10988_BASE_DIR}/tomograms/{ts_id}.rec"
    slice_of_interest = empiar10988_ts_to_slice_of_interest[ts_id]
    tomo = -1 * load_mrc_data(tomo_file).float()[slice_of_interest]  
    # save tomogram for preprocessing
    if not os.path.exists(f"{tmp_dir}/raw_data"):
        os.makedirs(f"{tmp_dir}/raw_data")
    save_mrc_data(tomo, f"{tmp_dir}/raw_data/{ts_id}.mrc")

# preprocess test data
## step 1: create preprocessing config file
cfg_dir = os.path.dirname(train_cfg_file)
pre_config_file = f"{cfg_dir}/preprocess_test.py"
lines = [
    "pre_config={",
    f"\"dset_name\": \"empiar10988_preprocess\",",
    f"\"base_path\": \"{tmp_dir}\",",
    f"\"tomo_path\": \"{tmp_dir}/raw_data\",",
    f"\"tomo_format\": \".mrc\",",
    f"\"norm_type\": \"standardization\",",
    f"\"skip_coords\": \"{True}\",",  # don't need particle coordinates for testing
    f"\"skip_labels\": \"{True}\",",  # don't need labels for testing
    f"\"skip_ocp\": \"{True}\"",  # don't need occupancy for testing
    "}"
]
## step 2: save preprocessing config file
if os.path.exists(pre_config_file):
    os.remove(pre_config_file)
with open(pre_config_file, "w") as f:
    for line in lines:
        f.write(line + "\n")
## step 3: run preprocessing
os.system(f"python ./DeepETPicker_ProPicker/bin/preprocess.py --pre_configs {pre_config_file}")

# modify train config file to re-use it as test config file
## step 1: load train config file
module_name = "train_configs_module"  # You can pick any module name you like
spec = importlib.util.spec_from_file_location(module_name, train_cfg_file)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
## step 2: modify train config file
test_configs = copy.deepcopy(train_module.train_configs)
test_configs["pre_configs"] = pre_config_file
test_configs["train_set_ids"] = f"0-{len(test_ts)-1}"
test_configs["val_set_ids"] = f"0-{len(test_ts)-1}"
test_configs["gpu_ids"] = str(gpu)  # string is needed for gpu as multiple gpus can be used
test_configs["batch_size"] = batch_size
# also need this info from preprocessing cfg
test_configs["dset_name"] = "test"
test_configs["base_path"] = tmp_dir
test_configs["tomo_path"] = f"{tmp_dir}/raw_data"
## step 3: save test config file
test_cfg_file = f"{cfg_dir}/test.py"
if os.path.exists(test_cfg_file):
    os.remove(test_cfg_file)
with open(test_cfg_file, "w") as f:
    f.write("train_configs=")
    # write test_configs as string with double quotes
    f.write(str(test_configs).replace("'", "\""))
        
# inference
os.system(
    f"python ./DeepETPicker_ProPicker/bin/test_bash.py \
        --train_configs {test_cfg_file} \
        --checkpoints {ckpt_file} \
        --de_duplication True \
        --network ProPicker \
        --propicker_model_file '{PROPICKER_MODEL_FILE}' \
        --prompt_embed_file './fixed_prompts_empiar10988.json' \
    "
)

# remove raw data
shutil.rmtree(f"{tmp_dir}")
