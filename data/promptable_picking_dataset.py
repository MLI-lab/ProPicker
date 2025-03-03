#%%
import sys
sys.path.append("..")
sys.path.append("../..")

from utils.mrctools import * 
from data.utils import *
import torch
from torch.utils.data import Dataset
import glob
import pandas as pd
import torch
import h5py
import os
import json
import numpy as np
import random
from .utils import load_h5_tensor_dict

class PromptablePickingDataset(Dataset):
    def __init__(self, model_input_files, model_target_files, max_classes_per_tomo=torch.inf, normalization="gaussian", prompt_type="tomotwin_temb_files", prompt_dict_json=None, fixed_prompts=False, limit_toclasses=None, add_background_class=False, augmentation_pipeline=None, name=None):
        # self.run_dirs = run_dirs
        self.model_target_files = model_target_files
        self.model_input_files = model_input_files
        self.max_classes_per_tomo = max_classes_per_tomo
        self.normalization = normalization
        assert prompt_type in ["tomotwin_temb_files", "dict"], f"Prompt type {prompt_type} not implemented"
        if prompt_type == "dict":
            assert prompt_dict_json is not None, "Prompt dict json must be provided if prompt type is dict"
            if not fixed_prompts:
                print("Warning: fixed_prompts=False has no efect when setting prompt_type=dict. All prompts will be fixed dict entries!")       
        self.prompt_type = prompt_type
        self.fixed_prompts = fixed_prompts
        self.limit_toclasses = limit_toclasses
        self.add_background_class = add_background_class
        self.prompt_dict_json = prompt_dict_json
        self.prompt_dict = self._load_prompt_dict()    
        self.name=name
        self.augmentation_pipeline = augmentation_pipeline

    
    def sample_template_prompts(self, prompt_dir, classes, index=None):
        prompts = []
        for cls in classes:
            try:
                embed_df = pd.read_pickle(f"{prompt_dir}/{cls}/embeddings.temb")
            except FileNotFoundError:
                embed_df = pd.read_pickle(f"{prompt_dir}/{cls.upper()}/embeddings.temb")
            # select random row            
            if self.fixed_prompts:
                row = embed_df.iloc[0]
            else:
                row = embed_df.sample()
            subtomo_embed = row[[str(k) for k in range(0, 32)]].values.astype(np.float32)
            subtomo_embed = torch.from_numpy(subtomo_embed).squeeze()
            prompts.append(subtomo_embed)
        prompts = torch.stack(prompts)
        return prompts, row.index[0]

    def _load_prompt_dict(self):
        if self.prompt_type != "dict":
            return None
        else:
            with open(self.prompt_dict_json, "r") as f:
                prompt_dict = json.load(f)
            prompt_dict = {key.lower(): torch.tensor(value) for key, value in prompt_dict.items()}
            return prompt_dict

    def normalize(self, x):
        if self.normalization is None:
            return x
        elif self.normalization == "gaussian":
            return (x - x.mean()) / x.std()
        else:
            raise ValueError(f"Normalization {self.normalization} not implemented")

    def __len__(self):
        return len(self.model_target_files)

    def _check_if_in_limit_toclasses(self, cls):
        if self.limit_toclasses is None:
            return True
        else:
            return cls in self.limit_toclasses

    def __getitem__(self, index):
        model_input_file = self.model_input_files[index]
        model_input = load_mrc_data(model_input_file)   
        model_target_file = self.model_target_files[index]     
        model_target_dict = load_h5_tensor_dict(model_target_file)
        model_target_dict = dict(sorted(model_target_dict.items(), reverse=True))
        model_target_dict = {k: v for k, v in model_target_dict.items() if self._check_if_in_limit_toclasses(k)}
        
        if len(model_target_dict) > self.max_classes_per_tomo:
            keys_sample = random.sample(list(model_target_dict.keys()), self.max_classes_per_tomo)
            model_target_dict = {k: model_target_dict[k] for k in keys_sample}
        
        if self.add_background_class:
            target_sum = torch.stack(list(model_target_dict.values())).sum(dim=0)
            background = torch.ones_like(target_sum) - target_sum
            # insert background at the beginning which is important for MONAI implementation of DiceLosses
            model_target_dict_items = list(model_target_dict.items())
            model_target_dict_items.insert(0, ("background", background))
            model_target_dict = dict(model_target_dict_items)
        
        classes = list(model_target_dict.keys())
        
        prompt_dir = model_target_file.split("/subtomos/locmaps/")[0] + "/prompts/embeds"
        if self.prompt_type == "tomotwin_temb_files":
            prompts, prompt_index = self.sample_template_prompts(prompt_dir, classes, index=index)
        elif self.prompt_type == "dict":
            prompts = torch.stack([self.prompt_dict[cls.lower()] for cls in classes if cls.lower() in self.prompt_dict.keys()])
            prompt_index = torch.zeros(len(classes))
        
        model_targets = torch.stack(list(model_target_dict.values()))
        num_pos_voxels_per_target = model_targets.sum(dim=(1,2,3))
        item = {
            "model_input": self.normalize(model_input),
            "model_targets": model_targets,
            "num_pos_voxels_per_target": num_pos_voxels_per_target,
            "prompts": prompts,
            "classes": classes,
            "model_input_file": model_input_file,
            "model_target_file": model_target_file,
            "prompt_index": prompt_index,
        }
        if self.augmentation_pipeline is not None:
            item = self.augmentation_pipeline(item)
        return item



# ALL_TOMOTWINclasses = [
#     "1avo",
#     "1fzg",
#     "1jpm",
#     "2hmi",
#     "2vyr",
#     "3ewf",
#     "1e9r",
#     "1oao",
#     "2df7",
#     "5xnl",
#     "1ul1",
#     "2rhs",
#     "3mkq",
#     "7ey7",
#     "3ulv",
#     "1n9g",
#     "7blq",
#     "6wzt",
#     "7egq",
#     "5vkq",
#     "7lsy",
#     "7kdv",
#     "6lxv",
#     "7dd9",
#     "7amv",
#     "7nhs",
#     "7e8h",
#     "7e1y",
#     "2ww2",
#     "7vtq",
#     "6yt5",
#     "7egd",
#     "7sn7",
#     "7woo",
#     "7mei",
#     "7t3u",
#     "6z6o",
#     "7bkc",
#     "7eep",
#     "7e8s",
#     "7qj0",
#     "7nyz",
#     "6vqk",
#     "6ziu",
#     "6x02",
#     "7e6g",
#     "7o01",
#     "6x5z",
#     "7wbt",
#     "6vgr",
#     "4uic",
#     "6z3a",
#     "7kfe",
#     "7wi6",
#     "7shk",
#     "5tzs",
#     "7ege",
#     "7ETM",
#     "6SCJ",
#     "6tav",
#     "2vz9",
#     "6klh",
#     "1kb9",
#     "3pxi",
#     "4ycz",
#     "6igc",
#     "6f8l",
#     "6JY0",
#     "6TA5",
#     "6TGC",
#     "2dfs",
#     "6ksp",
#     "7jsn",
#     "6KRK",
#     "7niu",
#     "5a20",
#     "5ool",
#     "6up6",
#     "6i0d",
#     "6bq1",
#     "7SFW",
#     "3lue",
#     "6jk8",
#     "5h0s",
#     "6lx3",
#     "5ljo",
#     "6duz",
#     "4xk8",
#     "6xf8",
#     "6M04",
#     "6u8q",
#     "6lxk",
#     "6CE7",
#     "5csa",
#     "7sgm",
#     "7b5s",
#     "6gym",
#     "6emk",
#     "6w6m",
#     "7r04",
#     "5o32",
#     "6ces",
#     "2xnx",
#     "6LMT",
#     "7blr",
#     "2r9r",
#     "6zqj",
#     "4wrm",
#     "7s7k",
#     "4V94",
#     "4CR2",
#     "1QVR",
#     "1BXN",
#     "3CF3",
#     "1U6G",
#     "3D2F",
#     "2CG9",
#     "3H84",
#     "3GL1",
#     "3QM1",
#     "1S3X",
#     "5MRC",
#     "1FPY",
#     "1FO4",
#     "1FZ8",
#     "1JZ8",
#     "4ZIT",
#     "5BK4",
#     "5BW9",
#     "1CU1",
#     "1SS8",
#     "6AHU",
#     "6TPS",
#     "6X9Q",
#     "6GY6",
#     "6NI9",
#     "6VZ8",
#     "4HHB",
#     "7B7U",
#     "6Z80",
#     "6PWE",
#     "6PIF",
#     "6O9Z",
#     "6ID1",
#     "5YH2",
#     "4RKM",
#     "1G3I",
#     "1DGS",
#     "1CLQ",
#     "7Q21",
#     "7KJ2",
#     "7K5X",
#     "7FGF",
#     "7CRQ",
#     "6YBS",
#     "5JH9",
#     "5A8L",
#     "3IF8",
#     "2B3Y",
#     "6VN1",
#     "6MRC",
#     "6CNJ",
#     "5G04",
#     "4QF6",
#     "1SVM",
#     "1O9J",
#     "1ASZ",
# ]
# %%
