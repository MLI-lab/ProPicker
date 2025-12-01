import pytorch_lightning as pl
from torch import nn

from propicker.paths import SHREC2021_BASE_DIR, TOMOTWIN_MODEL_FILE, TOMOTWIN_TOMO_BASE_DIR

# these are passed to the preparation function of all datasets 
shared_preparation_function_kwargs = {
    "subtomo_size": 64,
    "subtomo_extraction_strides": [32, 32, 32],
    "crop_tomo_fn": lambda x: x,  # use this to crop tomo before extracting subtomos
    "save_full_locmaps": False,
    "setup_tomotwin_reference_embeddings": True,  
    "tomotwin_model_file": TOMOTWIN_MODEL_FILE,  # path to the tomotwin model file used as prompt encoder
    "skip_existing": True,  # if true, the preparation function will skip existing files
    "device": "cuda:0",  # some preparation steps (e.g. extracting prompt embeddings) are faster on GPU
}
dataset_configs = {
    "tomotwin": {
        "preparation_function": "propicker.data.preparation_functions.prepare_tomotwin.prepare_tomotwin_run",
        "preparation_function_kwargs": {
            "dataset_base_dir": TOMOTWIN_TOMO_BASE_DIR,
            **shared_preparation_function_kwargs
        },
        "train_val_items": [
            "tomo_simulation_round_1/tomo_01.2022-04-11T140327+0200",
            "tomo_simulation_round_1/tomo_02.2022-04-11T140327+0200",
            "tomo_simulation_round_1/tomo_03.2022-04-11T140327+0200",
            "tomo_simulation_round_1/tomo_04.2022-04-11T170238+0200",
            "tomo_simulation_round_1/tomo_05.2022-04-11T170238+0200",
            "tomo_simulation_round_1/tomo_06.2022-04-11T170238+0200",
            "tomo_simulation_round_1/tomo_07.2022-04-11T191718+0200",

            "tomo_simulation_round_2/tomo_01.2022-05-10T161313+0200",
            "tomo_simulation_round_2/tomo_02.2022-05-10T161313+0200",
            "tomo_simulation_round_2/tomo_03.2022-05-10T161313+0200",
            "tomo_simulation_round_2/tomo_04.2022-05-10T182221+0200",
            "tomo_simulation_round_2/tomo_05.2022-05-10T182221+0200",
            "tomo_simulation_round_2/tomo_06.2022-05-10T182221+0200",
            "tomo_simulation_round_2/tomo_07.2022-05-10T200254+0200",

            "tomo_simulation_round_3/tomo_01.2022-04-12T084242+0200",
            "tomo_simulation_round_3/tomo_02.2022-04-12T084242+0200",
            "tomo_simulation_round_3/tomo_03.2022-04-12T084242+0200",
            "tomo_simulation_round_3/tomo_04.2022-04-12T110927+0200",
            "tomo_simulation_round_3/tomo_05.2022-04-12T110927+0200",
            "tomo_simulation_round_3/tomo_06.2022-04-12T110927+0200",
            "tomo_simulation_round_3/tomo_07.2022-04-12T130359+0200",
               
            "tomo_simulation_round_5/tomo_01.2022-05-11T105658+0200", 
            "tomo_simulation_round_5/tomo_02.2022-05-11T105658+0200",
            "tomo_simulation_round_5/tomo_03.2022-05-11T105658+0200",
            "tomo_simulation_round_5/tomo_04.2022-05-11T123052+0200",
            "tomo_simulation_round_5/tomo_05.2022-05-11T123052+0200",
            "tomo_simulation_round_5/tomo_06.2022-05-11T123052+0200",
            "tomo_simulation_round_5/tomo_07.2022-05-11T134419+0200",

            "tomo_simulation_round_6/tomo_01.2022-04-12T165107+0200",
            "tomo_simulation_round_6/tomo_02.2022-04-12T165107+0200",
            "tomo_simulation_round_6/tomo_03.2022-04-12T165107+0200",
            "tomo_simulation_round_6/tomo_04.2022-04-12T192044+0200",
            "tomo_simulation_round_6/tomo_05.2022-04-12T192044+0200",
            "tomo_simulation_round_6/tomo_06.2022-04-12T192044+0200",
            "tomo_simulation_round_6/tomo_07.2022-04-12T211730+0200",

            "tomo_simulation_round_7/tomo_01.2022-05-12T084041+0200",
            "tomo_simulation_round_7/tomo_02.2022-05-12T084041+0200",
            "tomo_simulation_round_7/tomo_03.2022-05-12T084041+0200",
            "tomo_simulation_round_7/tomo_04.2022-05-12T113036+0200",
            "tomo_simulation_round_7/tomo_05.2022-05-12T113036+0200",
            "tomo_simulation_round_7/tomo_06.2022-05-12T113036+0200",
            "tomo_simulation_round_7/tomo_07.2022-05-12T140747+0200",

            "tomo_simulation_round_8/tomo_01.2022-05-12T095215+0200",
            "tomo_simulation_round_8/tomo_02.2022-05-12T095215+0200",
            "tomo_simulation_round_8/tomo_03.2022-05-12T095215+0200",
            "tomo_simulation_round_8/tomo_04.2022-05-12T110720+0200",
            "tomo_simulation_round_8/tomo_05.2022-05-12T110720+0200",
            "tomo_simulation_round_8/tomo_06.2022-05-12T110720+0200",
            "tomo_simulation_round_8/tomo_07.2022-05-12T121323+0200",

            "tomo_simulation_round_9/tomo_01.2022-04-12T172855+0200",
            "tomo_simulation_round_9/tomo_02.2022-04-12T172855+0200",
            "tomo_simulation_round_9/tomo_03.2022-04-12T172855+0200",
            "tomo_simulation_round_9/tomo_04.2022-04-12T193813+0200",
            "tomo_simulation_round_9/tomo_05.2022-04-12T193813+0200",
            "tomo_simulation_round_9/tomo_06.2022-04-12T193813+0200",
            "tomo_simulation_round_9/tomo_07.2022-04-12T212347+0200",

            "tomo_simulation_round_10/tomo_01.2022-05-03T143345+0200",
            "tomo_simulation_round_10/tomo_02.2022-05-03T143345+0200",
            "tomo_simulation_round_10/tomo_03.2022-05-03T143345+0200",
            "tomo_simulation_round_10/tomo_04.2022-05-03T183726+0200",
            "tomo_simulation_round_10/tomo_05.2022-05-03T183726+0200",
            "tomo_simulation_round_10/tomo_06.2022-05-03T183726+0200",
            "tomo_simulation_round_10/tomo_07.2022-05-03T213239+0200",

            "tomo_simulation_round_11/tomo_01.2022-05-04T093702+0200",
            "tomo_simulation_round_11/tomo_02.2022-05-04T093702+0200",
            "tomo_simulation_round_11/tomo_03.2022-05-04T093702+0200",
            "tomo_simulation_round_11/tomo_04.2022-05-04T114207+0200",
            "tomo_simulation_round_11/tomo_05.2022-05-04T114207+0200",
            "tomo_simulation_round_11/tomo_06.2022-05-04T114207+0200",
            "tomo_simulation_round_11/tomo_07.2022-05-04T132658+0200",
        ],
        "exclusive_val_items": [
            "tomo_simulation_round_1/tomo_08.2022-04-11T191718+0200",
            "tomo_simulation_round_2/tomo_08.2022-05-10T200254+0200",
            "tomo_simulation_round_3/tomo_08.2022-04-12T130359+0200",
            "tomo_simulation_round_5/tomo_08.2022-05-11T134419+0200",
            "tomo_simulation_round_6/tomo_08.2022-04-12T211730+0200",
            "tomo_simulation_round_7/tomo_08.2022-05-12T140747+0200",
            "tomo_simulation_round_8/tomo_08.2022-05-12T121323+0200",
            "tomo_simulation_round_9/tomo_08.2022-04-12T212347+0200",
            "tomo_simulation_round_10/tomo_08.2022-05-03T213239+0200",
            "tomo_simulation_round_11/tomo_08.2022-05-04T132658+0200",
        ],
    },
    "shrec2021": {
        "preparation_function": "propicker.data.preparation_functions.prepare_shrec2021.prepare_shrec2021_model",
        "preparation_function_kwargs": {
            "dataset_base_dir": SHREC2021_BASE_DIR,
            **shared_preparation_function_kwargs
        },
        "train_val_items": [
            "model_0",
            "model_1",
            "model_2",
            "model_3",
            "model_4",
            "model_5",
            "model_6",
            "model_7",
        ],
        "exclusive_val_items": [
            "model_8"
        ],
    },
}
datamodule_args = {
    "dataset_configs": dataset_configs,
    "output_base_dir": f"./datasets/propicker_training_subtomos",  # directory where the datamodule will save the subtomos for training, WARNING: this requires a lot of disk space
    "limit_to_classes": None,
    "max_classes_per_tomo": 8,
    
    # this config ensures that for each particle a prompt is randomly sampled 
    "prompt_type": "tomotwin_temb_files",
    "fixed_prompts": False,
    
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

f_maps = [24, 48, 96, 2*96]  # deepetip default
f_maps = [int(f*4.5) for f in f_maps]  # scale up f_maps
model_args = {
    "encoder_args": 
        {
            "class": "model.encoder.ResidualUNet3DEncoder",
            "class_args": {
                "f_maps": f_maps,   # default
                "in_channels": 1, 
                "norm": "in",  # default in options
                "act": "relu",  # default in options
                "use_IP": True,  # Image Pyramid; default in train_bash
                "use_coord": True,  # Coordinate Convolution; default in train_bash
                "use_lw": False,  # LightWeight; default in train_bash
                "lw_kernel": 3,  # default in train_bash
            },
        },
    "decoder_args": 
        {
            "class": "model.promptable_decoder.ResidualUNet3DDecoder",
            "class_args": {
                "f_maps": f_maps,   # default
                "out_channels": 1, 
                "norm": "in",  # default in options
                "act": "relu",  # default in options
                "use_coord": True,  # Coordinate Convolution; default in train_bash
                "use_softmax": False,  # my defaults for single class
                "use_sigmoid": True,  # my defaults for single class
                "use_lw": False,  # LightWeight; default in train_bash
                "lw_kernel": 3,  # default in train_bash
                "promptable": True, 
                "prompt_dim": 32,
                "film_activation": None,
            },
        },
}

train_loss_args = val_loss_args = {
    "class": "torch.nn.BCELoss",
    "class_args": {
        "reduction": 'none',
    },
    "mask_empty_targets": True,  # ignore batch elements with empty targets
}

optimizer_args = {
    "class": "torch.optim.Adam",
    "class_args": {
        "lr": 1e-2,
    }
}
scheduler_args = None
trainer_args = { 
    "gpus": [0],
    "max_epochs": 10,
    "num_sanity_val_steps": 2,
    "check_val_every_n_epoch": 1,
    "detect_anomaly": True,
}



logdir = "./lightning_logs"
logger_name = "propicker_training"
logger = pl.loggers.TensorBoardLogger(logdir, name=logger_name)
