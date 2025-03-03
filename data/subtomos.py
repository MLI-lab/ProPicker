import math

import numpy as np
import tqdm
import torch
import tempfile
import os
import glob
import copy

class _Subtomos(torch.utils.data.Dataset):
    def __init__(self, tomo, subtomo_start_coords, subtomo_size):
        super().__init__()
        self.tomo = copy.deepcopy(tomo)
        self.subtomo_start_coords = subtomo_start_coords
        self.subtomo_size = subtomo_size

    def __len__(self):
        return len(self.subtomo_start_coords)
    
    def __getitem__(self, idx):
        start_coords = self.subtomo_start_coords[idx]
        subtomo = self.tomo[
            start_coords[0] : start_coords[0] + self.subtomo_size,
            start_coords[1] : start_coords[1] + self.subtomo_size,
            start_coords[2] : start_coords[2] + self.subtomo_size,
        ]
        return subtomo

def extract_subtomos(
    tomo,
    subtomo_size,
    subtomo_extraction_strides=None,
    enlarge_subtomos_for_rotating=False,
    pad_before_subtomo_extraction=False,
    return_pad_tomo_shape=False,
):
    """
    Extracts sub-tomograms of size 'subtomo_size' using a 3D sliding window approach. The three strides of the sliding window are specified by 'subtomo_extraction_strides', which must be three integers.
    If 'enlarge_subtomos_for_rotating' is True, sub-tomograms are extracted with shape sqrt(2)*'subtomo_size, so they can be rotated and cropped to 'subtomo_size' without zero-filling.
    """
    # TODO: refactor subtomo_extraction_strides to subtomo_overlap
    if enlarge_subtomos_for_rotating:
        subtomo_size = ceil_to_even_integer(math.sqrt(2) * subtomo_size)
    if subtomo_extraction_strides is None:
        subtomo_extraction_strides = 3 * [subtomo_size]
    if pad_before_subtomo_extraction:
        # pad for subtomo extraction with extraction strides
        pad_x = subtomo_extraction_strides[0] - (
            (tomo.shape[0] - subtomo_size) % subtomo_extraction_strides[0]
        )
        pad_y = subtomo_extraction_strides[1] - (
            (tomo.shape[1] - subtomo_size) % subtomo_extraction_strides[1]
        )
        pad_z = subtomo_extraction_strides[2] - (
            (tomo.shape[2] - subtomo_size) % subtomo_extraction_strides[2]
        )
        # pad = torch.nn.ConstantPad3d((0, pad_z, 0, pad_y, 0, pad_x), 0)  # right pad with zero
        pad = torch.nn.ReflectionPad3d((0, pad_z, 0, pad_y, 0, pad_x))
        tomo = pad(tomo.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    # Generating starting indices for each subtomo
    subtomo_start_coords = [
        (i, j, k)
        for i in range(
            0, tomo.shape[0] - subtomo_size + 1, subtomo_extraction_strides[0]
        )
        for j in range(
            0, tomo.shape[1] - subtomo_size + 1, subtomo_extraction_strides[1]
        )
        for k in range(
            0, tomo.shape[2] - subtomo_size + 1, subtomo_extraction_strides[2]
        )
    ]
    subtomos = _Subtomos(tomo, subtomo_start_coords, subtomo_size)        
    # else:
    #     subtomos = (
    #         tomo.unfold(0, subtomo_size, subtomo_extraction_strides[0])
    #         .unfold(1, subtomo_size, subtomo_extraction_strides[1])
    #         .unfold(2, subtomo_size, subtomo_extraction_strides[2])
    #     )
    #     subtomos = subtomos.reshape(-1, subtomo_size, subtomo_size, subtomo_size)
    #     subtomos = list(subtomos)
    if return_pad_tomo_shape:
        return subtomos, subtomo_start_coords, tomo.shape   
    return subtomos, subtomo_start_coords


# def reassemble_subtomos(
#     subtomos, subtomo_start_coords, subtomo_overlap=None, crop_to_size=None
# ):
#     """
#     Basically the inverse of 'extract_subtomos'. For this to work, 'extract_subtomos' must have been called with 'pad_before_subtomo_extraction=True', and 'crop_to_size' must be set to the 3D shape of the tomogram from which the sub-tomograms were extracted.
#     """
#     # calculate the max indices in each dimension to infer the shape of the original tomogram
#     subtomo_size = subtomos[0].shape[0]
#     max_idx = [
#         max(start_idx[i] + subtomo_size for start_idx in subtomo_start_coords)
#         for i in range(3)
#     ]
#     if subtomo_overlap is None:
#         subtomo_weights = torch.ones_like(subtomos[0])
#     else:
#         subtomo_weights = get_linear_ramp_weights(
#             subtomos[0].shape[0], subtomo_overlap
#         ).to(subtomos[0].device)

#     out_vol = torch.zeros(max_idx, dtype=torch.float32, device=subtomos[0].device)
#     count_vol = torch.zeros_like(out_vol)
#     for subtomo, start_idx in zip(subtomos, subtomo_start_coords):
#         end_idx = [start + subtomo_size for start in start_idx]
#         out_vol[
#             start_idx[0] : end_idx[0],
#             start_idx[1] : end_idx[1],
#             start_idx[2] : end_idx[2],
#         ] += (
#             subtomo * subtomo_weights
#         )
#         count_vol[
#             start_idx[0] : end_idx[0],
#             start_idx[1] : end_idx[1],
#             start_idx[2] : end_idx[2],
#         ] += subtomo_weights
#     # avoid division by zero by replacing zero counts with ones
#     # count_vol[count_vol == 0] = 1
#     # average the overlapping regions by dividing the accumulated values by their count
#     out_vol /= count_vol
#     if crop_to_size is not None:
#         out_vol = out_vol[: crop_to_size[0], : crop_to_size[1], : crop_to_size[2]]
#     return out_vol

def _insert_subtomo(
    tomo, count_vol, subtomo, subtomo_start_coords, subtomo_weights, subtomo_overlap=None
):
    """
    insert a subtomogram into a volume, accumulating the values in 'tomo' and 'count_vol' and updating the count in 'count_vol'.
    """
    subtomo_size = subtomo.shape[0]
    end_idx = [start + subtomo_size for start in subtomo_start_coords]
    tomo[
        subtomo_start_coords[0] : end_idx[0],
        subtomo_start_coords[1] : end_idx[1],
        subtomo_start_coords[2] : end_idx[2],
    ] += (
        subtomo * subtomo_weights
    )
    count_vol[
        subtomo_start_coords[0] : end_idx[0],
        subtomo_start_coords[1] : end_idx[1],
        subtomo_start_coords[2] : end_idx[2],
    ] += subtomo_weights


def get_linear_ramp_weights(subtomo_size, subtomo_overlap, zero_border=0, ramp_end=None):
    """
    Produces a cubic 3D tensor containing linear weights used to average overlapping sub-tomogram parts in 'reassemble_subtomos'.
    """
    if ramp_end is None:
        ramp_end = subtomo_overlap

    if zero_border > ramp_end:
        raise ValueError(
            f"zero_border ({zero_border}) must be less than or equal to subtomo_overlap ({ramp_end})"
        )


    ramp_region = ramp_end - zero_border

    ramp = np.linspace(0, 1, ramp_region) + 1e-6
    
    weight_map_1d = np.ones(subtomo_size)

    if zero_border > 0:
        weight_map_1d[:zero_border] = 1e-6
        weight_map_1d[-zero_border:] = 1e-6
        weight_map_1d[zero_border:zero_border+ramp_region] = ramp  # Apply sigmoid ramp at the start
        weight_map_1d[-ramp_region-zero_border:-zero_border] = ramp[::-1]  # and at the end, inverted
    else:
        weight_map_1d[:ramp_region] = ramp
        weight_map_1d[-ramp_region:] = ramp[::-1]

    # Create a 3D weight map by extending the 1D weight map to 3 dimensions
    weight_map_3d = np.ones((subtomo_size, subtomo_size, subtomo_size))
    for i in range(subtomo_size):
        for j in range(subtomo_size):
            for k in range(subtomo_size):
                weight_map_3d[i, j, k] = (
                    weight_map_1d[i] * weight_map_1d[j] * weight_map_1d[k]
                )

    return torch.from_numpy(weight_map_3d)



def ceil_to_even_integer(x):
    """
    Produces the smallest even integer i that satisfies i >= x.
    """
    return int(math.ceil(x / 2.0) * 2)
