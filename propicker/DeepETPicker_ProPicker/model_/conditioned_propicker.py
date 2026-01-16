import importlib.util
import torch
import json
import pytorch_lightning as pl
from propicker.model.promptable_picker import ProPicker


class ConditionedProPicker(pl.LightningModule):
    def __init__(self, model, prompts):
        super().__init__()
        self.model = model
        self.prompts = prompts
        if not len(prompts.shape) == 2:
            raise ValueError("Prompts tensor must have shape (num_prompts, prompt_dim)")
    
    def forward(self, x):
        if not len(x.shape) == 5:
            raise ValueError("Input tensor must have shape (batch_size, num_channels, height, width, depth)")
        # repeat prompt along batch dimension
        prompts = self.prompts.repeat(x.shape[0], 1, 1) 
        batch = {
            "model_input": x.squeeze(1),
            "prompts": prompts
        }
        model_output = self.model.get_model_output(batch)
        return model_output
    

def load_conditioned_propicker(args):
    # load ProPicker model class from packaged propicker
    ppicker = ProPicker.load_from_checkpoint(args.propicker_model_file)
    
    # load prompts
    prompts = json.load(open(args.prompt_embed_file, "r"))
    if args.prompt_class is None or args.prompt_class == "None":
        if len(prompts.keys()) == 1:
            prompt_class = list(prompts.keys())[0]
        else:
            raise ValueError(f"Multiple classes (keys) found in prompt_embed_file file, please specify which one to use with --prompt_class")
    else:
        prompt_class = args.prompt_class

    prompts = torch.tensor(prompts[prompt_class]).unsqueeze(0)
    model = ConditionedProPicker(ppicker, prompts)    
    return model
