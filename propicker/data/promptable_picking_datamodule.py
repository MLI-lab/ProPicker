#%%
import math
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from .promptable_picking_dataset import PromptablePickingDataset

#%%
class PromptablePickingDatamodule(pl.LightningDataModule):
    def __init__(self, dataset_configs, output_base_dir, max_classes_per_tomo=torch.inf, prompt_type="tomotwin_reference_embedding", prompt_dict_json=None, fixed_prompts=False, limit_to_classes=None, add_background_class=False, val_frac=0.2, train_batch_size=1, val_batch_size=1, augmentation_pipeline=None, num_workers=0, seed=42):
        super().__init__()
        self.dataset_configs = dataset_configs
        self.output_base_dir = output_base_dir
        
        # self.tomotwin_tomo_base_dir = tomotwin_tomo_base_dir
        # self.train_val_tomotwin_runs = train_val_tomotwin_runs
        # self.exclusive_val_tomotwin_runs = exclusive_val_tomotwin_runs
        # self.all_tomotwin_runs = train_val_tomotwin_runs + exclusive_val_tomotwin_runs

        # self.shrec2021_base_dir = shrec2021_base_dir        
        # self.train_val_shrec2021_models = train_val_shrec2021_models
        # self.exclusive_val_shrec2021_models = exclusive_val_shrec2021_models
        # self.all_shrec2021_models = train_val_shrec2021_models + exclusive_val_shrec2021_models

        # self.shrec2020_base_dir = shrec2020_base_dir
        # self.train_val_shrec2020_models = train_val_shrec2020_models
        # self.exclusive_val_shrec2020_models = exclusive_val_shrec2020_models
        # self.all_shrec2020_models = train_val_shrec2020_models + exclusive_val_shrec2020_models

        # self.shrec2019_base_dir = shrec2019_base_dir
        # self.train_val_shrec2019_models = train_val_shrec2019_models
        # self.exclusive_val_shrec2019_models = exclusive_val_shrec2019_models
        # self.all_shrec2019_models = train_val_shrec2019_models + exclusive_val_shrec2019_models
        
        # self.empiar10988_base_dir = empiar10988_base_dir
        # self.train_val_empiar10988_ts = train_val_empiar10988_ts
        # self.exclusive_val_empiar10988_ts = exclusive_val_empiar10988_ts
        # self.all_empiar10988_ts = train_val_empiar10988_ts + exclusive_val_empiar10988_ts
        
        
        self.max_classes_per_tomo = max_classes_per_tomo
        
        self.prompt_type = prompt_type
        self.prompt_dict_json = prompt_dict_json
        self.fixed_prompts = fixed_prompts
        self.limit_to_classes = limit_to_classes
        self.add_background_class = add_background_class
        
        self.augmentation_pipeline = augmentation_pipeline
    
        self.val_frac = val_frac
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.seed = seed


    def prepare_data(self):
        # these lists will contain paths to input and target files and are later used to setup the datasets
        self.train_val_input_files, self.train_val_target_files, self.exclusive_val_input_files, self.exclusive_val_target_files = [], [], [], []
        for dataset_name, config in self.dataset_configs.items():
            self._prepare_dataset(
                dataset_name,
                config,
            )


    def _prepare_dataset(self, dataset_name, dataset_config):
        # apply preparation function to items
        preparation_function = eval(dataset_config["preparation_function"])
        train_val_items = dataset_config["train_val_items"]
        exclusive_val_items = dataset_config["exclusive_val_items"]

        print(f"Preparing {len(train_val_items) + len(exclusive_val_items)} tomos from dataset '{dataset_name}'")
        for item in train_val_items + exclusive_val_items:
            is_exclusive = item in exclusive_val_items
            item_output_dir = f"{self.output_base_dir}/{dataset_name}/{item}"

            preparation_function(
                item=item,
                out_dir=item_output_dir,
                **dataset_config["preparation_function_kwargs"],
                # subtomo_extraction_strides=self.subtomo_extraction_strides,
                # save_full_locmaps=True,
                # tomotwin_model_file=tomotwin_model_file,
                # setup_tomotwin_reference_embeddings=setup_tomotwin_reference_embeddings,
                # skip_existing=skip_existing,
                # crop_tomo_fn=crop_tomo_fn,
            )

            for k in range(len(os.listdir(f"{item_output_dir}/subtomos/model_inputs"))):
                input_file = f"{item_output_dir}/subtomos/model_inputs/{k}.mrc"
                target_file = f"{item_output_dir}/subtomos/locmaps/{k}.h5"
                if is_exclusive:
                    self.exclusive_val_input_files.append(input_file)
                    self.exclusive_val_target_files.append(target_file)
                else:
                    self.train_val_input_files.append(input_file)
                    self.train_val_target_files.append(target_file)

    # def prepare_data(self, setup_tomotwin_reference_embeddings=True, tomotwin_model_file=None, skip_existing=True, crop_tomo_fn=None) -> None:
    #     """
    #     Saves sub tomograms for model training and validation on disk. Must be called before train and val loaders can be accessed.
    #     """
    #     if setup_tomotwin_reference_embeddings and tomotwin_model_file is None:
    #         raise ValueError("Must provide path to tomotwin model file if setting up tomotwin reference embeddings.")
    #     # these lists will contain paths to input and target files and are later used to setup the datasets
    #     self.train_val_input_files, self.train_val_target_files = [], []
    #     self.exclusive_val_input_files, self.exclusive_val_target_files = [], []
    #     if len(self.all_shrec2021_models) > 0: 
    #         self._prepare_shrec2021_data(
    #             setup_tomotwin_reference_embeddings=setup_tomotwin_reference_embeddings, 
    #             tomotwin_model_file=tomotwin_model_file,
    #             crop_tomo_fn=crop_tomo_fn,
    #             skip_existing=skip_existing
    #         )
    #     if len(self.all_shrec2020_models) > 0:
    #         self._prepare_shrec2020_data(
    #             setup_tomotwin_reference_embeddings=setup_tomotwin_reference_embeddings, 
    #             tomotwin_model_file=tomotwin_model_file,
    #             crop_tomo_fn=crop_tomo_fn,
    #             skip_existing=skip_existing
    #         )
    #     if len(self.all_shrec2019_models) > 0:
    #         self._prepare_shrec2019_data(
    #             setup_tomotwin_reference_embeddings=setup_tomotwin_reference_embeddings, 
    #             tomotwin_model_file=tomotwin_model_file,
    #             crop_tomo_fn=crop_tomo_fn,
    #             skip_existing=skip_existing
    #         )
    #     if len(self.all_tomotwin_runs) > 0:
    #         self._prepare_tomotwin_data(
    #             setup_tomotwin_reference_embeddings=setup_tomotwin_reference_embeddings, 
    #             tomotwin_model_file=tomotwin_model_file,
    #             crop_tomo_fn=crop_tomo_fn,
    #             skip_existing=skip_existing
    #     )
    #     if len(self.all_empiar10988_ts) > 0:
    #         self._prepare_empiar10988_data(
    #             setup_tomotwin_reference_embeddings=setup_tomotwin_reference_embeddings, 
    #             tomotwin_model_file=tomotwin_model_file,
    #             crop_tomo_fn=crop_tomo_fn,
    #             skip_existing=skip_existing
    #         )

    # def _prepare_tomotwin_data(self, setup_tomotwin_reference_embeddings, skip_existing, tomotwin_model_file, crop_tomo_fn):
    #     for run in self.all_tomotwin_runs:
    #         tomotwin_run_dir = f"{self.tomotwin_tomo_base_dir}/{run}"
    #         output_run_dir = f"{self.output_base_dir}/{run}"
    #         setup_segmentation_subtomos_tomotwin_run(
    #             run_dir=tomotwin_run_dir,
    #             out_dir=output_run_dir,
    #             subtomo_size=self.subtomo_size,
    #             subtomo_extraction_strides=self.subtomo_extraction_strides,
    #             setup_tomotwin_reference_embeddings=setup_tomotwin_reference_embeddings,
    #             tomotwin_model_file=tomotwin_model_file, 
    #             skip_existing=skip_existing,
    #             save_full_locmaps=True,
    #             crop_tomo_fn=crop_tomo_fn, 
    #         )         
    #         # append paths to all input and target files we have just created to the respective lists
    #         for k in range(len(os.listdir(f"{output_run_dir}/subtomos/model_inputs"))):
    #             if any([run in output_run_dir for run in self.exclusive_val_tomotwin_runs]):
    #                 self.exclusive_val_input_files.append(f"{output_run_dir}/subtomos/model_inputs/{k}.mrc")
    #                 self.exclusive_val_target_files.append(f"{output_run_dir}/subtomos/locmaps/{k}.h5")
    #             else:
    #                 self.train_val_input_files.append(f"{output_run_dir}/subtomos/model_inputs/{k}.mrc")
    #                 self.train_val_target_files.append(f"{output_run_dir}/subtomos/locmaps/{k}.h5")
            
    # def _prepare_shrec2021_data(self, setup_tomotwin_reference_embeddings, skip_existing, tomotwin_model_file, crop_tomo_fn):
    #     for model in self.all_shrec2021_models:
    #         shrec_model_dir = f"{self.shrec2021_base_dir}/{model}"
    #         output_model_dir = f"{self.output_base_dir}/shrec2021_{model}"
    #         setup_segmentation_tomos_shrec2021_model(
    #             shrec_model_dir=shrec_model_dir, 
    #             out_dir=output_model_dir, 
    #             subtomo_size=self.subtomo_size, 
    #             subtomo_extraction_strides=self.subtomo_extraction_strides,
    #             setup_tomotwin_reference_embeddings=setup_tomotwin_reference_embeddings,
    #             tomotwin_model_file=tomotwin_model_file, 
    #             skip_existing=skip_existing,
    #             save_full_locmaps=True,
    #             crop_tomo_fn=crop_tomo_fn, 
    #         )
    #         for k in range(len(os.listdir(f"{output_model_dir}/subtomos/model_inputs"))):
    #             if any([model in output_model_dir for model in self.exclusive_val_shrec2021_models]):
    #                 self.exclusive_val_input_files.append(f"{output_model_dir}/subtomos/model_inputs/{k}.mrc")
    #                 self.exclusive_val_target_files.append(f"{output_model_dir}/subtomos/locmaps/{k}.h5")
    #             else:
    #                 self.train_val_input_files.append(f"{output_model_dir}/subtomos/model_inputs/{k}.mrc")
    #                 self.train_val_target_files.append(f"{output_model_dir}/subtomos/locmaps/{k}.h5")

    # def _prepare_shrec2020_data(self, setup_tomotwin_reference_embeddings, skip_existing, tomotwin_model_file, crop_tomo_fn):
    #     for model in self.all_shrec2020_models:
    #         shrec_model_dir = f"{self.shrec2020_base_dir}/{model}"
    #         output_model_dir = f"{self.output_base_dir}/shrec2020_{model}"
    #         setup_segmentation_tomos_shrec2020_model(
    #             shrec_model_dir=shrec_model_dir, 
    #             out_dir=output_model_dir, 
    #             subtomo_size=self.subtomo_size, 
    #             subtomo_extraction_strides=self.subtomo_extraction_strides,
    #             setup_tomotwin_reference_embeddings=setup_tomotwin_reference_embeddings,
    #             tomotwin_model_file=tomotwin_model_file, 
    #             skip_existing=skip_existing,
    #             save_full_locmaps=True,
    #             crop_tomo_fn=crop_tomo_fn, 
    #         )
    #         for k in range(len(os.listdir(f"{output_model_dir}/subtomos/model_inputs"))):
    #             if any([model in output_model_dir for model in self.exclusive_val_shrec2020_models]):
    #                 self.exclusive_val_input_files.append(f"{output_model_dir}/subtomos/model_inputs/{k}.mrc")
    #                 self.exclusive_val_target_files.append(f"{output_model_dir}/subtomos/locmaps/{k}.h5")
    #             else:
    #                 self.train_val_input_files.append(f"{output_model_dir}/subtomos/model_inputs/{k}.mrc")
    #                 self.train_val_target_files.append(f"{output_model_dir}/subtomos/locmaps/{k}.h5")

    # def _prepare_shrec2019_data(self, setup_tomotwin_reference_embeddings, skip_existing, tomotwin_model_file, crop_tomo_fn):
    #     for model in self.all_shrec2019_models:
    #         shrec_model_dir = f"{self.shrec2019_base_dir}/{model}"
    #         output_model_dir = f"{self.output_base_dir}/shrec2019_{model}"
    #         setup_segmentation_tomos_shrec2019_model(
    #             shrec_model_dir=shrec_model_dir, 
    #             out_dir=output_model_dir, 
    #             subtomo_size=self.subtomo_size, 
    #             subtomo_extraction_strides=self.subtomo_extraction_strides,
    #             setup_tomotwin_reference_embeddings=setup_tomotwin_reference_embeddings,
    #             tomotwin_model_file=tomotwin_model_file, 
    #             skip_existing=skip_existing,
    #             save_full_locmaps=True,
    #             crop_tomo_fn=crop_tomo_fn, 
    #         )
    #         for k in range(len(os.listdir(f"{output_model_dir}/subtomos/model_inputs"))):
    #             if any([model in output_model_dir for model in self.exclusive_val_shrec2019_models]):
    #                 self.exclusive_val_input_files.append(f"{output_model_dir}/subtomos/model_inputs/{k}.mrc")
    #                 self.exclusive_val_target_files.append(f"{output_model_dir}/subtomos/locmaps/{k}.h5")
    #             else:
    #                 self.train_val_input_files.append(f"{output_model_dir}/subtomos/model_inputs/{k}.mrc")
    #                 self.train_val_target_files.append(f"{output_model_dir}/subtomos/locmaps/{k}.h5")
                    
    # def _prepare_empiar10988_data(self, setup_tomotwin_reference_embeddings, skip_existing, tomotwin_model_file, crop_tomo_fn):
    #     for ts in self.all_empiar10988_ts:
    #         output_ts_dir = f"{self.output_base_dir}/empiar10988_{ts}"
    #         setup_segmentation_subtomos_empiar10988_ts(
    #             empiar10988_base_dir=self.empiar10988_base_dir,
    #             empiar10988_ts_id=ts,
    #             out_dir=output_ts_dir,
    #             subtomo_size=self.subtomo_size,
    #             subtomo_extraction_strides=self.subtomo_extraction_strides,
    #             save_full_locmaps=True,
    #             tomotwin_model_file=tomotwin_model_file,
    #             setup_tomotwin_reference_embeddings=setup_tomotwin_reference_embeddings,
    #             skip_existing=skip_existing,
    #             crop_tomo_fn=crop_tomo_fn,
    #         )
    #         for k in range(len(os.listdir(f"{output_ts_dir}/subtomos/model_inputs"))):
    #             if any([ts in output_ts_dir for ts in self.exclusive_val_empiar10988_ts]):
    #                 self.exclusive_val_input_files.append(f"{output_ts_dir}/subtomos/model_inputs/{k}.mrc")
    #                 self.exclusive_val_target_files.append(f"{output_ts_dir}/subtomos/locmaps/{k}.h5")
    #             else:
    #                 self.train_val_input_files.append(f"{output_ts_dir}/subtomos/model_inputs/{k}.mrc")
    #                 self.train_val_target_files.append(f"{output_ts_dir}/subtomos/locmaps/{k}.h5")
                    

    def setup(self, stage=None):
        # randomly sample val_runs_frac of runs for validation
        train_val_dataset = PromptablePickingDataset(
            model_input_files=self.train_val_input_files,
            model_target_files=self.train_val_target_files,
            prompt_type=self.prompt_type,
            prompt_dict_json=self.prompt_dict_json,
            max_classes_per_tomo=self.max_classes_per_tomo,
            fixed_prompts=self.fixed_prompts,
            limit_toclasses=self.limit_to_classes,
            add_background_class=self.add_background_class,
            augmentation_pipeline=self.augmentation_pipeline,
        )
        n_val = int(math.ceil(len(train_val_dataset) * self.val_frac))
        n_train = len(train_val_dataset) - n_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            train_val_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(self.seed)
        )
        self.exclusive_val_dataset = PromptablePickingDataset(
            model_input_files=self.exclusive_val_input_files,
            model_target_files=self.exclusive_val_target_files,
            prompt_type=self.prompt_type,
            prompt_dict_json=self.prompt_dict_json,
            max_classes_per_tomo=self.max_classes_per_tomo,
            limit_toclasses=self.limit_to_classes,
            add_background_class=self.add_background_class,
            fixed_prompts=self.fixed_prompts,
            name="exclusive",
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
           batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        if len(self.exclusive_val_input_files) == 0:
            return val_dataloader
        else:
            exclusive_val_dataloader = DataLoader(
                self.exclusive_val_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
            )
            return val_dataloader, exclusive_val_dataloader
