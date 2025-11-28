# ProPicker

## Update
We now ship CLI/GUI entry points. See `GUI_QUICKSTART.md` for the end‑to‑end workflow.

## Installation and Setup
We recommend using Conda to install the necessary dependencies. To do so, run the following commands:
```
conda env create -f environment.yml
conda activate ppicker
```

To use ProPicker, you need the checkpoint of our pre-trained model, as well as the checkpoint of the TomoTwin model we used as prompt encoder: 

- You can download the ProPicker checkpoint here [here](https://drive.google.com/file/d/1lFDIJdAc99QVfGDuhOu89vEXYlz-oHQh/view?usp=share_link)

- You can download the TomoTwin checkpoint by running `bash download_tomotwin_ckpt.sh`

After downloading, place the files in the `ProPicker` directory. If you want to store them somewhere else, adjust the paths in `propicker/paths.py`.

## Prompt-Based Picking 
We provide an example for prompt-based picking in the `TUTORIAL1:empiar10988_prompt_based_picking.ipynb` notebook, in which we pick ribosomes in the EMPIAR-10988 dataset. 


## Fine-Tuning ProPicker
An example for fine-tuning ProPicker on the EMPIAR-10988 dataset is provided in the `TUTORIAL2:empiar10988_fine_tuning.ipynb` notebook.

## Training ProPicker from Scratch

Training from scratch lives in `propicker/training_from_scratch/train.py`, with parameters in `propicker/training_from_scratch/train_cfg.py`.

To download the training data, you can use `datasets/download_train_data.sh`.

**Note:** The training data is large, so you might want to download it to a different location. To do this, modify `datasets/download_train_data.sh`; also adjust the training data path in `propicker/paths.py`.


## Note
This repository contains code copied and modified from the following projects:
- [DeepETPicker](https://github.com/cbmi-group/DeepETPicker)
- [DeePiCt](https://github.com/ZauggGroup/DeePiCt)
- [TomoTwin](https://github.com/MPI-Dortmund/tomotwin-cryoet)

All derived code is explicitly marked as such.
