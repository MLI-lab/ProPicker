#%%
from propicker.data.subtomos import extract_subtomos, _insert_subtomo, get_linear_ramp_weights
import tqdm
import torch
from torch.utils.data import DataLoader
from .tomotwin import get_tomotwin_prompt_embeds_dict

#%%
def get_pred_locmap_dict(model, tomo, prompt_embeds_dict=None, prompt_subtomos_dict=None, tomotwin_model_file=None, subtomo_size=64, subtomo_overlap=32, batch_size=1, mean_pool_locmaps=0, subtomo_normalization="gaussian", zero_border=0, ramp_end=None, num_dataloader_workers=1):
    ## STEP 1: embed prompts if necessary
    if prompt_embeds_dict is not None and prompt_subtomos_dict is not None:
        raise ValueError("'prompt_embeds_dict' and 'prompt_subtomos_dict' cannot both be specified!")
    if prompt_subtomos_dict is not None:
        if tomotwin_model_file is None:
            raise ValueError("tomotwin_model_file must be specified if 'prompt_subtomos_dict' is not None!")
        prompt_embeds_dict = get_tomotwin_prompt_embeds_dict(
            prompt_subtomos_dict=prompt_subtomos_dict,
            tomotwin_model_file=tomotwin_model_file,
            batch_size=batch_size,
            device=model.device,
        )
        torch.cuda.empty_cache()
    ## STEP 2: Pass subtomos through model
    # prepare subtomos
    subtomo_dataset, subtomos_coords, pad_tomo_shape = extract_subtomos(
        tomo=tomo,#.to(model.device),
        subtomo_size=subtomo_size,
        subtomo_extraction_strides=3*[subtomo_size-subtomo_overlap],
        pad_before_subtomo_extraction=True,
        return_pad_tomo_shape=True
    )
    pred_locmap_dict = {
        cls: torch.zeros(pad_tomo_shape, device=model.device)
        for cls in prompt_embeds_dict.keys()
    }
    count_dict = {
        cls: torch.zeros(pad_tomo_shape, device=model.device)
        for cls in prompt_embeds_dict.keys()
    }
    # process subtomos
    subtomo_loader = DataLoader(
        subtomo_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_dataloader_workers,
        #pin_memory=True,
        drop_last=False
    )
    if any([prompt is None for prompt in prompt_embeds_dict.values()]):
        if not all([prompt is None for prompt in prompt_embeds_dict.values()]):
            raise ValueError("either all or none of the prompts should be None")
        prompts = None
    else:
        prompts = torch.stack(list(prompt_embeds_dict.values()))#.to(model.device)
    subtomo_weights = get_linear_ramp_weights(
        subtomo_size=subtomo_size, subtomo_overlap=subtomo_overlap, zero_border=zero_border, ramp_end=ramp_end
    ).to(model.device)
    with torch.no_grad():
        for batch_idx, subtomo_batch in tqdm.tqdm(enumerate(subtomo_loader), total=len(subtomo_loader), desc=f"Inference ({len(pred_locmap_dict)} prompts)"):
            if subtomo_normalization is None:
                pass
            elif subtomo_normalization == "gaussian":
                subtomo_batch -= subtomo_batch.mean(dim=(-1,-2,-3),
                    keepdim=True)
                subtomo_batch /= subtomo_batch.std(dim=(-1,-2,-3),
                    keepdim=True)
            batch = {
                "model_input": subtomo_batch,
                "prompts": prompts.repeat(subtomo_batch.shape[0], 1, 1) if prompts is not None else None,
            }
            # this is messy
            if hasattr(model, "get_model_outputs"):
                batch_out = model.get_model_outputs(batch)
            elif hasattr(model, "get_model_output"):
                batch_out = model.get_model_output(batch)
            else:
                raise ValueError("model should have either get_model_outputs or get_model_output method")
            for k in range(len(prompt_embeds_dict)):
                cls = list(prompt_embeds_dict.keys())[k]
                pred_locmap = pred_locmap_dict[cls]
                count_vol = count_dict[cls]
                batch_out_class = batch_out[:,k,...]   
                for batch_sample_idx, subtomo in enumerate(batch_out_class):
                    _insert_subtomo(
                        tomo=pred_locmap,
                        count_vol=count_vol,
                        subtomo=subtomo,
                        subtomo_start_coords=subtomos_coords[batch_idx*batch_size + batch_sample_idx],
                        subtomo_weights=subtomo_weights,
                        subtomo_overlap=subtomo_overlap
                    )
        for cls, pred_locmap in pred_locmap_dict.items():
            pred_locmap = pred_locmap / count_dict[cls]
            pred_locmap = pred_locmap[:tomo.shape[0], :tomo.shape[1], :tomo.shape[2]]
            if mean_pool_locmaps > 0:
                pred_locmap = mean_pool(pred_locmap, num=mean_pool_locmaps)
            pred_locmap_dict[cls] = pred_locmap.cpu()
    return pred_locmap_dict


def mean_pool(locmap, kernel=3, num=5):
    """
    This function is inspired by the _nms_v2 routine used in DeepETPicker:
    https://github.com/cbmi-group/DeepETPicker/blob/main/test.py
    """
    locmap_ = locmap.clone()    
    if locmap.ndim == 3:
        locmap_ = locmap.unsqueeze(0)
    elif locmap.ndim == 4 or locmap.ndim == 5:
        pass
    else:
        raise ValueError("locmap should be 3D (possibly with additional batch and/or channel dimensions)")
    with torch.no_grad():
        #locmap_ = torch.where(locmap_ > 0.5, 1, 0)
        meanPool = torch.nn.AvgPool3d(kernel, 1, kernel // 2).to(locmap.device)
        maxPool = torch.nn.MaxPool3d(kernel, 1, kernel // 2).to(locmap.device)
        hmax = locmap_.clone().float()
        for _ in range(num):
            hmax = meanPool(hmax)    
        locmap_ = hmax.clone()
        locmap_ = maxPool(locmap_)
    locmap_ = locmap_.squeeze(0)
    return locmap_
