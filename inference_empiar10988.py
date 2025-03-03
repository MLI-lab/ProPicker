#%%

from matplotlib import pyplot as plt
import torch

from clustering_and_picking import get_cluster_centroids_df, get_best_case_cluster_based_picking_performance 
from paths import EMPIAR10988_BASE_DIR, TOMOTWIN_MODEL_FILE, PROPICKER_MODEL_FILE
from data.preparation_functions.prepare_empiar10988 import read_empiar10988_coords, empiar10988_ts_to_slice_of_interest
from data.utils import draw_balls
from inference import get_pred_locmap_dict
from train import ProPicker
from utils.mrctools import *

#%%
ts_id = "TS_030"

tomo_file = f"{EMPIAR10988_BASE_DIR}/tomograms/{ts_id}.rec"
ribo_coord_file = f"{EMPIAR10988_BASE_DIR}/particle_lists/{ts_id}_cyto_ribosomes.csv"


#%%
# tomograms contain large emtpy regions, so we only consider the interesting part
slice_of_interest = empiar10988_ts_to_slice_of_interest[ts_id]
tomo = -1 * load_mrc_data(tomo_file).float()
tomo = tomo[slice_of_interest].clone()


coords = read_empiar10988_coords(ribo_coord_file)
coords.Z -= slice_of_interest.start

plt.imshow(tomo.sum(1))


#%%
# extract all ribosomes
all_ribo_subtomos = []
for coord in coords[["X", "Y", "Z"]].values.astype(int):
    x, y, z = coord
    subtomo = tomo[
        z-18:z+19,
        y-18:y+19,
        x-18:x+19
    ]
    if not subtomo.shape == (37, 37, 37):
        continue
    all_ribo_subtomos.append(subtomo)
# choose a single ribosome as prompt
prompt_subtomos_dict = {"cyto_ribosome": all_ribo_subtomos[268]}
plt.imshow(prompt_subtomos_dict["cyto_ribosome"][18])

#%%
subtomo_size = 64
subtomo_overlap = 32
batch_size = 64
device = "cuda:3"

model = ProPicker.load_from_checkpoint(PROPICKER_MODEL_FILE).to(device)
model = model.to(device).eval()
model.freeze()
pred_locmap_dict = get_pred_locmap_dict(
    model, 
    tomo, 
    prompt_subtomos_dict=prompt_subtomos_dict, 
    tomotwin_model_file=TOMOTWIN_MODEL_FILE,
    subtomo_size=subtomo_size, 
    subtomo_overlap=subtomo_overlap, 
    batch_size=batch_size
)

#%%
# these parameters are used for cluster-based picking and have to be tuned manually for each dataset
binarization_thresh = 0.5
min_cluster_size = 50
max_cluster_size = 4/3 * torch.pi * (37/2)**3  # choose slightly larget than the particle size

cluster_cenroids = get_cluster_centroids_df(
    binary_locmap=pred_locmap_dict["cyto_ribosome"] > binarization_thresh,
)
cluster_cenroids_filt = cluster_cenroids[
    (min_cluster_size <= cluster_cenroids["size"]) & (cluster_cenroids["size"] <= max_cluster_size)
]

#%%
# draw balls around cluster centroids
picks = draw_balls(
    positions=cluster_cenroids_filt,
    shape=tomo.shape,
    radius=9,
    device=device,
)

#%%
gt_picks = load_mrc_data(f"{EMPIAR10988_BASE_DIR}/labels/{ts_id}_cyto_ribosomes.mrc")[slice_of_interest].bool()

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(tomo[100])
ax[0].imshow(picks[100], alpha=0.2)
ax[1].imshow(tomo[100])
ax[1].imshow(gt_picks[100], alpha=0.2)

#%%    
# add dimensions of bounding boxes for true positive calculation
coords["height"] = coords["width"] = coords["depth"] = 37
best_stats = get_best_case_cluster_based_picking_performance(
    pred_locmap_dict=pred_locmap_dict,
    optimize_thresh=True,
    n_size_steps=5,
    n_thresh_steps=10,
    gt_positions=coords,
    metric="F1",
    num_workers=0,
    iou_thresh=0.6,  # a prediction is considered a true positive if the IoU is above this threshold
)       

# %%
