import pytorch_lightning as pl
from torch import nn
from paths import TOMOTWIN_TOMO_BASE_DIR, SHREC2021_BASE_DIR, TRAIN_VAL_SUBTOMOS_DIR, TOMOTWIN_MODEL_FILE

subtomo_size = 64
max_epochs = 10 
gpus = [1]

datamodule_args = {
    "class_args": {
        "tomotwin_tomo_base_dir": TOMOTWIN_TOMO_BASE_DIR,
        # which tomotwin data to use for trianing, we exclude tomo 8 and use it as test set 
        "train_val_tomotwin_runs": 
            [
                # "tomo_simulation_round_1/tomo_01.2022-04-11T140327+0200",
                # "tomo_simulation_round_1/tomo_02.2022-04-11T140327+0200",
                # "tomo_simulation_round_1/tomo_03.2022-04-11T140327+0200",
                # "tomo_simulation_round_1/tomo_04.2022-04-11T170238+0200",
                # "tomo_simulation_round_1/tomo_05.2022-04-11T170238+0200",
                # "tomo_simulation_round_1/tomo_06.2022-04-11T170238+0200",
                # "tomo_simulation_round_1/tomo_07.2022-04-11T191718+0200",

                # "tomo_simulation_round_2/tomo_01.2022-05-10T161313+0200",
                # "tomo_simulation_round_2/tomo_02.2022-05-10T161313+0200",
                # "tomo_simulation_round_2/tomo_03.2022-05-10T161313+0200",
                # "tomo_simulation_round_2/tomo_04.2022-05-10T182221+0200",
                # "tomo_simulation_round_2/tomo_05.2022-05-10T182221+0200",
                # "tomo_simulation_round_2/tomo_06.2022-05-10T182221+0200",
                # "tomo_simulation_round_2/tomo_07.2022-05-10T200254+0200",

                # "tomo_simulation_round_3/tomo_01.2022-04-12T084242+0200",
                # "tomo_simulation_round_3/tomo_02.2022-04-12T084242+0200",
                # "tomo_simulation_round_3/tomo_03.2022-04-12T084242+0200",
                # "tomo_simulation_round_3/tomo_04.2022-04-12T110927+0200",
                # "tomo_simulation_round_3/tomo_05.2022-04-12T110927+0200",
                # "tomo_simulation_round_3/tomo_06.2022-04-12T110927+0200",
                # "tomo_simulation_round_3/tomo_07.2022-04-12T130359+0200",
                
                # # exclude these tomos for evaluation and fine tuning experiments
                # # "tomo_simulation_round_4/tomo_01.2022-05-12T111913+0200",
                # # "tomo_simulation_round_4/tomo_02.2022-05-12T111913+0200",
                # # "tomo_simulation_round_4/tomo_03.2022-05-12T111913+0200",
                # # "tomo_simulation_round_4/tomo_04.2022-05-12T124440+0200",
                # # "tomo_simulation_round_4/tomo_05.2022-05-12T124440+0200",
                # # "tomo_simulation_round_4/tomo_06.2022-05-12T124440+0200",
                # # "tomo_simulation_round_4/tomo_07.2022-05-12T135012+0200",
                
                # "tomo_simulation_round_5/tomo_01.2022-05-11T105658+0200", 
                # "tomo_simulation_round_5/tomo_02.2022-05-11T105658+0200",
                # "tomo_simulation_round_5/tomo_03.2022-05-11T105658+0200",
                # "tomo_simulation_round_5/tomo_04.2022-05-11T123052+0200",
                # "tomo_simulation_round_5/tomo_05.2022-05-11T123052+0200",
                # "tomo_simulation_round_5/tomo_06.2022-05-11T123052+0200",
                # "tomo_simulation_round_5/tomo_07.2022-05-11T134419+0200",

                # "tomo_simulation_round_6/tomo_01.2022-04-12T165107+0200",
                # "tomo_simulation_round_6/tomo_02.2022-04-12T165107+0200",
                # "tomo_simulation_round_6/tomo_03.2022-04-12T165107+0200",
                # "tomo_simulation_round_6/tomo_04.2022-04-12T192044+0200",
                # "tomo_simulation_round_6/tomo_05.2022-04-12T192044+0200",
                # "tomo_simulation_round_6/tomo_06.2022-04-12T192044+0200",
                # "tomo_simulation_round_6/tomo_07.2022-04-12T211730+0200",

                # "tomo_simulation_round_7/tomo_01.2022-05-12T084041+0200",
                # "tomo_simulation_round_7/tomo_02.2022-05-12T084041+0200",
                # "tomo_simulation_round_7/tomo_03.2022-05-12T084041+0200",
                # "tomo_simulation_round_7/tomo_04.2022-05-12T113036+0200",
                # "tomo_simulation_round_7/tomo_05.2022-05-12T113036+0200",
                # "tomo_simulation_round_7/tomo_06.2022-05-12T113036+0200",
                # "tomo_simulation_round_7/tomo_07.2022-05-12T140747+0200",

                # "tomo_simulation_round_8/tomo_01.2022-05-12T095215+0200",
                # "tomo_simulation_round_8/tomo_02.2022-05-12T095215+0200",
                # "tomo_simulation_round_8/tomo_03.2022-05-12T095215+0200",
                # "tomo_simulation_round_8/tomo_04.2022-05-12T110720+0200",
                # "tomo_simulation_round_8/tomo_05.2022-05-12T110720+0200",
                # "tomo_simulation_round_8/tomo_06.2022-05-12T110720+0200",
                # "tomo_simulation_round_8/tomo_07.2022-05-12T121323+0200",

                # "tomo_simulation_round_9/tomo_01.2022-04-12T172855+0200",
                # "tomo_simulation_round_9/tomo_02.2022-04-12T172855+0200",
                # "tomo_simulation_round_9/tomo_03.2022-04-12T172855+0200",
                # "tomo_simulation_round_9/tomo_04.2022-04-12T193813+0200",
                # "tomo_simulation_round_9/tomo_05.2022-04-12T193813+0200",
                # "tomo_simulation_round_9/tomo_06.2022-04-12T193813+0200",
                # "tomo_simulation_round_9/tomo_07.2022-04-12T212347+0200",

                # "tomo_simulation_round_10/tomo_01.2022-05-03T143345+0200",
                # "tomo_simulation_round_10/tomo_02.2022-05-03T143345+0200",
                # "tomo_simulation_round_10/tomo_03.2022-05-03T143345+0200",
                # "tomo_simulation_round_10/tomo_04.2022-05-03T183726+0200",
                # "tomo_simulation_round_10/tomo_05.2022-05-03T183726+0200",
                # "tomo_simulation_round_10/tomo_06.2022-05-03T183726+0200",
                # "tomo_simulation_round_10/tomo_07.2022-05-03T213239+0200",

                # "tomo_simulation_round_11/tomo_01.2022-05-04T093702+0200",
                # "tomo_simulation_round_11/tomo_02.2022-05-04T093702+0200",
                # "tomo_simulation_round_11/tomo_03.2022-05-04T093702+0200",
                # "tomo_simulation_round_11/tomo_04.2022-05-04T114207+0200",
                # "tomo_simulation_round_11/tomo_05.2022-05-04T114207+0200",
                # "tomo_simulation_round_11/tomo_06.2022-05-04T114207+0200",
                # "tomo_simulation_round_11/tomo_07.2022-05-04T132658+0200",
            ],
        "exclusive_val_tomotwin_runs": 
            [], # runs specified here are exclusively used in a special validation set
        "shrec2021_base_dir": SHREC2021_BASE_DIR,
        "train_val_shrec2021_models": 
            [
                "model_0",
                # "model_1",
                # "model_2",
                # "model_3",
                # "model_4",
                # "model_5",
                # "model_6",
                # "model_7",
            ],
        "exclusive_val_shrec2021_models": 
            [],

        "train_val_subtomos_dir": TRAIN_VAL_SUBTOMOS_DIR, # training data is stored here

        "limit_to_classes": None,  # restrict training to these classes; useful for fine-tuning

        "subtomo_size": subtomo_size,
        "subtomo_extraction_strides": [32, 32, 32],  # extract subtomos with these strides
        "max_classes_per_tomo": 10,

        # this config ensures that for each particle a prompt is randomly sampled 
        "prompt_type": "tomotwin_reference_embedding",
        "prompt_dict_json": None,
        "fixed_prompts": False,

        "val_frac": 0.1,
        "train_batch_size": 8 if subtomo_size==37 else 4,
        "val_batch_size": 32 if subtomo_size==37 else 8,
        "num_workers": 8,
        "seed": 42,
    },
    "prepare_data_args": {
        "setup_tomotwin_reference_embeddings": True,
        "tomotwin_model_file": TOMOTWIN_MODEL_FILE,
        "skip_existing": True,
    },
}

model_args = {
    # 'siamese_net_3d_encoder_args' are the arguments for the TomomTwin encoder
    # I took the'siamese_net_3d_encoder_args' from 'best_f1_after600.pth' file that came with the tomotwin demo
    "encoder_args": 
        {
            "class": "my_tomotwin.modules.networks.SiameseNet3D.SiameseNet3D",
            "class_args": {
                "output_channels": 32,
                "dropout": 0.2,
                "repeat_layers": 0,
                "norm_name": "GroupNorm",
                "norm_kwargs": {"num_groups": 64, "num_channels": 1024},
                "gem_pooling_p": 0,
            },
        },
    "decoder_args": 
        {
            "class": "model.siamese_net_3d.promptable_decoder.SiameseNet3DDecoder",
            "class_args": {
                "out_size": 64,
                "prompt_dim": 32,
                "film_activation": None,
                "final_activation": nn.Sigmoid(),
                "out_chans": 1,
            },
        },
    # the promptable_decoder is my own design. it is supposed to be the decoder counterpart to the tomotwin encoder
    "freeze_encoder": False
}

train_loss_args = val_loss_args = {
    "class": "torch.nn.BCELoss",
    "class_args": {
        "reduction": 'none',
    },
    "mask_empty_targets": False,
}

optimizer_args = {
    "class": "torch.optim.Adam",
    "class_args": {
        "lr": 1e-2,
    }
}


augmentation_args = [
    {
        "class": "data.augmentation.Flip",
        "class_args": {
            "axis": (0, 1),
            "prob": 0.3,
        }
    },
    {
        "class": "data.augmentation.Flip",
        "class_args": {
            "axis": (0, 2),
            "prob": 0.3,
        }
    },
    {
        "class": "data.augmentation.Flip",
        "class_args": {
            "axis": (1, 2),
            "prob": 0.3,
        }
    },
    {
        "class": "data.augmentation.RotateFull",
        "class_args": {
            "axes": (1, 2),
            "seed": int(1e8),
            "prob": 0.3,
        }
    },
]

logdir = "./lightning_logs"
logger_name = "propicker"
logger = pl.loggers.TensorBoardLogger(logdir, name=logger_name)

coldstart = True