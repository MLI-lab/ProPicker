
import torch
from tomotwin.modules.networks.networkmanager import NetworkManager
import tqdm
import json
import pandas as pd

def load_tomotwin_model(weightspth, load_weights=True, device=None):
    """
    Adapted from tomotwin.modules.inference.embedor
    """
    if device is not None:
        checkpoint = torch.load(weightspth, map_location=device)
    else:
        checkpoint = torch.load(weightspth)
    tomotwin_config = checkpoint["tomotwin_config"]
    print("Model config:")
    print(tomotwin_config)
    model = NetworkManager.create_network(tomotwin_config).get_model()
    if load_weights:
        before_parallel_failed = False
        if checkpoint is not None:
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError:
                print("Load before failed")
                before_parallel_failed = True
        #model = torch.nn.DataParallel(model)
        if before_parallel_failed:
            model.load_state_dict(checkpoint["model_state_dict"])
        print("Successfully loaded model weights")
    else:
        print("Model weights not loaded")
    return model.to(device)

def pass_subtomos_through_tomotwin(subtomos, tomotwin_model_file, batch_size=32, device="cpu"):    
    if not torch.is_tensor(subtomos):
        raise ValueError("Subtomos must be a 4-dim torch tensor! Got type: ", type(subtomos))
    if not subtomos.ndim == 4:
        raise ValueError("Subtomos must be a 4-dim torch tensor! Input has dim: ", subtomos.ndim)
    if not subtomos.shape[-1] == subtomos.shape[-2] == subtomos.shape[-3] == 37:
        raise ValueError("Subtomos must be 37x37x37 for TomoTwin! Input has shape: ", subtomos.shape)
    # tomotwin was trained with mean 0 and std 1 subtomos
    subtomos -= subtomos.mean(dim=(-1, -2, -3), keepdim=True)
    subtomos /= subtomos.std(dim=(-1, -2, -3), keepdim=True)
    subtomos = subtomos.view(len(subtomos), 1, 37, 37, 37)
    loader = torch.utils.data.DataLoader(subtomos, batch_size=batch_size, shuffle=False)
    tt = load_tomotwin_model(weightspth=tomotwin_model_file, device=device).eval()

    with torch.no_grad():
        tt_outputs = []
        for batch in tqdm.tqdm(loader, "Passing subtomos through TomoTwin"):
            tt_output = tt(batch.to(device))
            if isinstance(tt_output, tuple):
                tt_output = tt_output[0].cpu()
            if not tt_output.shape[1] == 32:
                raise ValueError("Tomotwin output is not shape 32! Something went wrong!")
            tt_outputs.append(tt_output)
        tt_outputs = torch.cat(tt_outputs)
 
    return tt_outputs

def eval_tomotwin_on_mask(tomo, mask, tomotwin_model_file, batch_size, device="cpu"):
    subtomos = []
    tomo = torch.nn.functional.pad(tomo, (19, 19, 19, 19, 19, 19), mode="constant")
    coords = mask.nonzero() + 19
    for centre in coords:
        subtomos.append(tomo[centre[0]-19:centre[0]+18, centre[1]-19:centre[1]+18, centre[2]-19:centre[2]+18])
    subtomos = torch.stack(subtomos)
    tt_outputs = pass_subtomos_through_tomotwin(subtomos, tomotwin_model_file, batch_size, device)
    return tt_outputs


def get_tomotwin_prompt_embeds_dict(prompt_subtomos_dict, tomotwin_model_file, device, batch_size=1, out_file=None):
    prompt_embeds = pass_subtomos_through_tomotwin(
        subtomos=torch.stack(list(prompt_subtomos_dict.values())),
        tomotwin_model_file=tomotwin_model_file,
        batch_size=batch_size,
        device=device
    )
    prompt_embeds_dict = {cls: prompt_embeds[i] for i, cls in enumerate(prompt_subtomos_dict.keys())}
    if out_file is not None:
        prompt_embeds_dict_ = {k: v.tolist() for k, v in prompt_embeds_dict.items()}
        with open(out_file, "w") as f:
            json.dump(prompt_embeds_dict_, f, indent=4)
    return prompt_embeds_dict