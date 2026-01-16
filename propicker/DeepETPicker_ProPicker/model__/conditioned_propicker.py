import importlib.util
import torch
import json
import pytorch_lightning as pl
  
def import_class_from_path(class_name, file_path):
    """
    Import a class from a file path by dynamically changing sys.path
    """
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # get the class from the module
    cls = getattr(module, class_name)
    return cls


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
    # load ProPicker model class
    propicker_base_dir = __file__.split("/DeepETPicker_ProPicker/")[0]
    ModelClass = import_class_from_path(
        class_name="ProPicker",
        file_path=f"{propicker_base_dir}/model/promptable_picker.py"
    )
    ppicker = ModelClass.load_from_checkpoint(args.propicker_model_file)
    
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