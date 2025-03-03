#%%
import torch
from model import ProPicker
import copy

ck = torch.load("/mnt/hdd_pool_zion/users/simon/foundation/lightning_logs/fixed_bce/deepetpicker_unet/version_0/checkpoints/epoch=2.ckpt", map_location="cpu")
ck_new = copy.deepcopy(ck)
hparams = ck["hyper_parameters"]
hparams["model_args"]["encoder_args"]["class"] = "model.encoder.ResidualUNet3DEncoder"
hparams["model_args"]["decoder_args"]["class"] = "model.promptable_decoder.ResidualUNet3DDecoder"
ck_new["hyper_parameters"] = hparams
torch.save(ck_new, "./mod_propicker_deepetpicker_unet.ckpt")
mo = ProPicker.load_from_checkpoint("./mod_propicker_deepetpicker_unet.ckpt")


#%%
import torch
from model import ProPicker
import copy

ck = torch.load("/home/simon/promptable_seg_submission/mod_propicker.ckpt", map_location="cpu")
ck_new = copy.deepcopy(ck)
hparams = ck["hyper_parameters"]  
hparams["model_args"]["encoder_args"]["class"] = "model_bak.encoder.SiameseNet3D"
hparams["model_args"]["decoder_args"]["class"] = "model_bak.promptable_decoder.SiameseNet3DDecoder"
ck_new["hyper_parameters"] = hparams
torch.save(ck_new, "./mod_propicker.ckpt")
mo = ProPicker.load_from_checkpoint("./mod_propicker.ckpt")


