import h5py 
import pandas as pd
import torch
import mrcfile
import torch
import tqdm


def load_mrc_data(mrc_file):
    with mrcfile.open(mrc_file, permissive=True) as mrc:
        try:
            data = torch.from_numpy(mrc.data)
        except TypeError:
            data = torch.from_numpy(mrc.data.astype(float))
    return data

def save_mrc_data(data, mrc_file):
    with mrcfile.new(mrc_file, overwrite=True) as mrc:
        mrc.set_data(data.clone().numpy())

def save_h5_tensor_dict(tensor_dict, file):
    with h5py.File(file, "w") as f:
        for k, v in tensor_dict.items():
            if not torch.is_tensor(v):
                raise ValueError(f"Value for key {k} is not a tensor")
            f.create_dataset(k, data=v.numpy())

def load_h5_tensor_dict(h5_file):
    with h5py.File(h5_file, "r") as f:
        dct = {
            k: torch.from_numpy(f[k][:]) for k in f.keys()
        }
    return dct


def center_crop_tomo_to_shape(tomo, shape_after_cropping):
    if tomo.shape[0] == shape_after_cropping[0] and tomo.shape[1] == shape_after_cropping[1] and tomo.shape[2] == shape_after_cropping[2]:
        return tomo
    cropz = (tomo.shape[0] - shape_after_cropping[0])//2
    cropy = (tomo.shape[1] - shape_after_cropping[1])//2
    cropx = (tomo.shape[2] - shape_after_cropping[2])//2
    tomo_crop = tomo[cropz:-cropz, cropy:-cropy, cropx:-cropx]
    return tomo_crop

def draw_balls(positions, shape, radius, device="cuda:0"): 
        
    if device == "cpu":
        print("WARNING: Using CPU to generate locmaps is very slow, consider using a GPU")

    # check if positions is a dataframe
    if isinstance(positions, pd.DataFrame):
        if "x" in positions.columns and "y" in positions.columns and "z" in positions.columns:
            xyz = positions[["x", "y", "z"]].values
        elif "X" in positions.columns and "Y" in positions.columns and "Z" in positions.columns:
            xyz = positions[["X", "Y", "Z"]].values

    coord_grid = torch.meshgrid([
        torch.arange(shape[0], device=device),
        torch.arange(shape[1], device=device), 
        torch.arange(shape[2], device=device), 
    ])
    coord_grid = torch.stack(coord_grid, dim=-1)

    ballmap = torch.zeros(shape, device=device)
    radius = torch.tensor(radius, device=device)
    for x, y, z in tqdm.tqdm(xyz, total=len(xyz), desc=f"Drawing balls"):
        center = torch.tensor([z, y, x], device=device) 
        class_ball = (coord_grid - center).pow(2).sum(-1) < radius**2
        ballmap[class_ball] = 1
    return ballmap.cpu()