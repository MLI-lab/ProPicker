import numpy as np
import pytorch_lightning as pl
import torch

from matplotlib import pyplot as plt
from torch import nn

import model


class PromptablePicker(nn.Module):
    def __init__(self, encoder_args, decoder_args, freeze_encoder=False):
        super().__init__()
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.encoder = eval(encoder_args["class"])(**encoder_args["class_args"])
        self.decoder = eval(decoder_args["class"])(**decoder_args["class_args"])

        if freeze_encoder:
            self.freeze_encoder()
    
    def freeze_encoder(self):
        self.encoder = self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode(self, x):
        return self.encoder.forward(x)

    def decode(self, feats, prompt):
        out = self.decoder(
            x=feats[0],
            cats=feats[1:],
            prompt=prompt,
        )
        return out

    def forward(self, x, prompt):
        _, feats = self.encoder.forward(x)
        out = self.decode(feats, prompt)
        return out



class ProPicker(pl.LightningModule):
    def __init__(self, model_args, optimizer_args, train_loss_args, val_loss_args=None, scheduler_args=None):
        super().__init__()
        self.model = PromptablePicker(**model_args)
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        self.train_loss_args = train_loss_args
        self.val_loss_args = val_loss_args if val_loss_args is not None else train_loss_args
        self.train_loss_fn = eval(self.train_loss_args["class"])(**self.train_loss_args["class_args"])
        self.val_loss_fn = eval(self.val_loss_args["class"])(**self.val_loss_args["class_args"])
        self.save_hyperparameters()

    def forward(self, x, prompt):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        assert len(x.shape) == 5, f"Expected 5D input, got {x.shape}"
        assert len(prompt.shape) == 2, f"Expected 2D input, got {prompt.shape}"
        return self.model(x, prompt)


    def get_loss(self, model_output, model_targets, mode, return_per_class_losses=False):
        assert mode in ["train", "val"], f"mode must be 'train' or 'val', got {mode}"
        loss_args = self.train_loss_args if mode == "train" else self.val_loss_args
        mask_empty_targets = loss_args["mask_empty_targets"]    
        loss_fn = self.train_loss_fn if mode == "train" else self.val_loss_fn

        if isinstance(loss_fn, torch.nn.BCELoss):
            # check if model output is in 0-1
            if model_output.min() < 0 or model_output.max() > 1:
                print("Model output is not in 0-1 range. Got min, max: ", model_output.min(), model_output.max())
                model_output = model_output.clamp(0, 1)
            if model_targets.min() < 0 or model_targets.max() > 1:
                print("Model targets is not in 0-1 range. Got min, max: ", model_targets.min(), model_targets.max())
                model_targets = model_targets.clamp(0, 1)
        per_class_losses = loss_fn(model_output, model_targets.to(model_output.device))
        if per_class_losses.ndim == 2:
            pass
        else:
            per_class_losses = per_class_losses.mean(dim=(-1, -2, -3))
        if mask_empty_targets:
            if len(per_class_losses.shape) == 2:
                mask = model_targets.sum(dim=(-1,-2,-3)) > 0
                if mask.sum() == 0:
                    mask[0, 0] = True
            loss = per_class_losses[mask].mean()
        else:
            loss = per_class_losses.mean()
        return (loss, per_class_losses) if return_per_class_losses else loss
    
    def get_model_output(self, batch):
        _, feats = self.model.encode(batch["model_input"].unsqueeze(1).to(self.device))
        # TODO: this is messy. currently, I assume that if the target has 1 channel, we use prompting and that if it has more than 1 channel, we don't use prompting
        if batch["prompts"] is None:
            model_output = self.model.decode(feats, prompt=None)
        else:
            output = []
            prompts = batch["prompts"].to(self.device)
            for prompt_id in range(prompts.shape[1]):  # prompt is a tensor of shape (batch_size, num_prompts, prompt_dim)
                prompt = prompts[:, prompt_id]
                model_output = self.model.decode(feats, prompt=prompt)
                output.append(model_output)
            # concat along prompt dimension
            model_output = torch.cat(output, dim=1)  
        return model_output

    def training_step(self, batch, batch_idx):
        model_output = self.get_model_output(batch)
        loss = self.get_loss(
            model_output=model_output, 
            model_targets=batch["model_targets"], 
            mode="train"
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        model_outputs = self.get_model_output(batch)        
        loss, per_class_losses = self.get_loss(
            model_output=model_outputs, 
            model_targets=batch["model_targets"], 
            mode="val",
            return_per_class_losses=True
        )
        # EVERYTHING BELOW IS JUST LOGGING
        # get name of dataset corresponding to dataloader_idx for logging
        dataset = self.trainer.val_dataloaders[dataloader_idx].dataset
        dataset_name = dataset.name if hasattr(dataset, "name") else None
        log_prefix = f"val_loss{f'_{dataset_name}' if dataset_name is not None else ''}"
        # log mean los
        self.log(f"{log_prefix}/mean", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        # log per cls loss
        unique_classes = set(np.array(batch["classes"]).flatten())
        per_class_loss_dict = {cls: [] for cls in unique_classes}
        per_class_model_output_dict = {cls: [] for cls in unique_classes}
        per_class_model_target_dict = {cls: [] for cls in unique_classes}
        for class_id in range(len(batch["classes"])):
            for batch_id in range(len(batch["classes"][0])):
                cls = batch["classes"][class_id][batch_id]
                per_class_loss_dict[cls].append(per_class_losses[batch_id][class_id].item())
                per_class_model_output_dict[cls].append(model_outputs[batch_id][class_id])
                per_class_model_target_dict[cls].append(batch["model_targets"][batch_id][class_id])
                
        # log some example outputs adn targets
        if dataloader_idx == 1:
            if batch_idx % 3 == 0:
                for cls in unique_classes:
                    model_output = per_class_model_output_dict[cls][0].squeeze().cpu()
                    model_target = per_class_model_target_dict[cls][0].squeeze().cpu()
                    all_targets = torch.zeros_like(model_target)
                    for class_ in per_class_model_target_dict.keys():
                        if class_ == "background":
                            continue
                        all_targets += per_class_model_target_dict[class_][0].squeeze().cpu()
                    all_targets = all_targets > 0

                    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
                    fig.suptitle(f"class={cls}, loss={per_class_loss_dict[cls][0]}")
                    ax[0].imshow(model_output.sum(0))
                    ax[1].imshow(model_target.sum(0))
                    ax[2].imshow(all_targets.sum(0))
                    ax[0].set_title("Output")
                    ax[1].set_title("Target")
                    ax[2].set_title("Sum of all cls targets")
                    self.logger.experiment.add_figure(f"{log_prefix}_output/{cls}_{batch_idx}", fig)    

        per_class_loss_dict = {k: sum(v) / len(v) for k, v in per_class_loss_dict.items()}
        for cls, class_loss in per_class_loss_dict.items():        
            self.log(f"{log_prefix}/{cls}", class_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)           
        return loss
    
    def configure_optimizers(self):
        print(f"Configuring optimizer {self.optimizer_args['class']} with args {self.optimizer_args['class_args']}")
        optimizer = eval(self.optimizer_args["class"])(self.parameters(), **self.optimizer_args["class_args"])
        out = {"optimizer": optimizer}
        if self.scheduler_args is not None:
            print(f"Configuring scheduler {self.scheduler_args['class']} with args {self.scheduler_args['class_args']}")
            scheduler = eval(self.scheduler_args["class"])(optimizer, **self.scheduler_args["class_args"])
            out["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val_loss_exclusive/mean/dataloader_idx_1",
                "interval": "epoch",
            }
        return out
    
    # def lr_scheduler_step(self, scheduler, metric):
    #     print("Running lr_scheduler_step")
    #     if metric is None:
    #         scheduler.step()
    #     else:
    #         scheduler.step(metric)
