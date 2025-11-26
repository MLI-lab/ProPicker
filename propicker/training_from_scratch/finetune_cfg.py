#%%
import pytorch_lightning as pl
from propicker.paths import EMPIAR10988_BASE_DIR, PROPICKER_MODEL_FILE, TOMOTWIN_MODEL_FILE


## DATA CFG
# these are passed to the preparation function of all datasets 
shared_preparation_function_kwargs = {
    "subtomo_size": 64,
    "subtomo_extraction_strides": [32, 32, 32],
    "crop_tomo_fn": lambda x: x,  # use this to crop tomo before extracting subtomos
    "save_full_locmaps": False,
    "setup_tomotwin_reference_embeddings": False,  # we don't need these as we have fixed prompts for finetuning
    "tomotwin_model_file": TOMOTWIN_MODEL_FILE,
    "skip_existing": False,
    "device": f"cuda:3",
}
dataset_configs = {
    "empiar10988": {
        "preparation_function": "propicker.data.preparation_functions.prepare_empiar10988.prepare_empiar10988_ts",
        "preparation_function_kwargs": {
            "dataset_base_dir": EMPIAR10988_BASE_DIR,
            **shared_preparation_function_kwargs
        },
        "train_val_items": ["TS_029"],
        "exclusive_val_items": ["TS_030"],
    },
}
datamodule_args = {
    "dataset_configs": dataset_configs,
    "output_base_dir": f"./finetuning_data",
    "limit_to_classes": None,
    "max_classes_per_tomo": 10,
    
    # this ensures that fixed prompts are used for finetuning
    "prompt_type": "dict",
    # the "prompt_dict_json" contains fixed prompts (TomoTwin embeddings) for each class considered during fine-tuning
    # if you want to fine-tune on a new dataset, you have to make a "prompt_dict_json" file with the inference.tomotwin.get_tomotwin_prompt_embeds_dict method
    "prompt_dict_json": "/home/simon/promptable_seg_submission/fixed_prompts_empiar10988.json", 
    "fixed_prompts": True,
    
    "val_frac": 0.01,
    "train_batch_size": 4,
    "val_batch_size": 8,
    "num_workers": 0,
    "seed": 42,
}
augmentation_args = [
    {
        "class": "propicker.data.augmentation.Flip",
        "class_args": {
            "axis": (0, 1),
            "prob": 0.3,
        }
    },
    {
        "class": "propicker.data.augmentation.Flip",
        "class_args": {
            "axis": (0, 2),
            "prob": 0.3,
        }
    },
    {
        "class": "propicker.data.augmentation.Flip",
        "class_args": {
            "axis": (1, 2),
            "prob": 0.3,
        }
    },
    {
        "class": "propicker.data.augmentation.RotateFull",
        "class_args": {
            "axes": (1, 2),
            "seed": int(1e8),
            "prob": 0.3,
        }
    },
]

## MODEL AND TRAINING CFG
model_args = {
    "ckpt": PROPICKER_MODEL_FILE,  # load this checkpoint for finetuning
}
train_loss_args = val_loss_args = {
    "class": "torch.nn.BCELoss",
    "class_args": {
        "reduction": 'none',
    },
    "mask_empty_targets": True,
}
optimizer_args = {
    "class": "torch.optim.Adam",
    "class_args": {
        "lr": 1e-3,
        #"weight_decay": 1e-7,
    }
}
# scheduler_args = {
#     "class": "torch.optim.lr_scheduler.ReduceLROnPlateau",
#     "class_args": {
#         "mode": "min",
#         "factor": 0.5,
#         "patience": 3,
#         "threshold_mode": "rel",
#         "cooldown": 0,
#         "min_lr": 1e-6,
#         "verbose": True,
#     }
# }
scheduler_args = None
trainer_args = { 
    "gpus": [3],
    "max_epochs": 100,
    "num_sanity_val_steps": 2,
    "check_val_every_n_epoch": 1,
    "detect_anomaly": True,
}

## LOGGING CFG
logdir = "./lightning_logs"
logger_name = f"fine_tune/empiar10988/train_ts={','.join(dataset_configs['empiar10988']['train_val_items'])}/val_ts={','.join(dataset_configs['empiar10988']['exclusive_val_items'])}/"
print(f"Logging to {logdir}/{logger_name}")
logger = pl.loggers.TensorBoardLogger(logdir, name=logger_name)
