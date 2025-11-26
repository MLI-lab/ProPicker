#%%
import os

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from propicker.data.promptable_picking_datamodule import PromptablePickingDatamodule
from propicker.data.augmentation import AugmentationPipeline
from propicker.model import ProPicker
from propicker.training_from_scratch.train_cfg import (
    augmentation_args,
    datamodule_args,
    logger,
    model_args,
    optimizer_args,
    train_loss_args,
    trainer_args,
    val_loss_args,
)
# if you want to fine tune a model, comment out the line above and uncomment the line below
# from propicker.training_from_scratch.finetune_cfg import (
#     datamodule_args,
#     logger,
#     train_loss_args,
#     val_loss_args,
#     model_args,
#     optimizer_args,
#     augmentation_args,
#     trainer_args,
# )

# this may be not needed on some systems
os.environ["NCCL_P2P_LEVEL"] = "PXB"
    
#%%
if __name__ == "__main__":
    seed_everything(187)

    augmentation_pipeline = AugmentationPipeline(
        [eval(args["class"])(**args["class_args"]) for args in augmentation_args],
        seed=int(1e9)
    )

    data_module = PromptablePickingDatamodule(
        augmentation_pipeline = augmentation_pipeline,
        **datamodule_args,
    )
    # this extracts subtomos for training and saves them to disk, it also generates prompts for all particles using tomotwin
    data_module.prepare_data()
    data_module.setup()

    # if we want to continue training from a checkpoint, we need to load the model and optimizer from the checkpoint
    if "ckpt" in model_args.keys():
        print(f"Found ckpt in model_args: {model_args['ckpt']}, finetuning model from this checkpoint")
        model = ProPicker.load_from_checkpoint(model_args["ckpt"])
    else:
        model = ProPicker(
            model_args=model_args, 
            optimizer_args=optimizer_args,
            train_loss_args=train_loss_args,
            val_loss_args=val_loss_args,
        )

    # callback to periodically save latest model
    epoch_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{logger.save_dir}/{logger.name}/version_{logger.version}/checkpoints",
        filename="{epoch}",
        monitor="epoch",
        mode="max",
        verbose=True,
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    val_loss_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{logger.save_dir}/{logger.name}/version_{logger.version}/checkpoints/val_loss",
        filename="min_val_loss_{epoch}",
        monitor="val_loss/mean/dataloader_idx_0",
        verbose=True,
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
    )
    exclusive_val_loss_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{logger.save_dir}/{logger.name}/version_{logger.version}/checkpoints/val_loss",
        filename="min_exclusive_val_loss_{epoch}",
        monitor="val_loss_exclusive/mean/dataloader_idx_1",
        verbose=True,
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[epoch_callback, val_loss_callback, exclusive_val_loss_callback],
        **trainer_args,
    )
    # Train the model
    if "ckpt" in model_args.keys():
        trainer.validate(model=model, dataloaders=data_module.val_dataloader())
    trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())   
