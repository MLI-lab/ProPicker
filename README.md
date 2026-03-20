# ProPicker
This repo contains an implementation of ProPicker as described in our [paper](https://doi.org/10.1016/j.jsb.2026.108298).
The repo covers [prompt-based picking](https://github.com/MLI-lab/ProPicker/blob/main/python_tutorial/tutorial1%3Aprompt_based_picking/tutorial.ipynb) and [particle-specific fine-tuning](https://github.com/MLI-lab/ProPicker/blob/main/python_tutorial/tutorial2%3Afine_tuning/tutorial.ipynb). We also provide code to train ProPicker from scratch. 

## Installation and Setup
1. Use Conda to install the necessary dependencies in a new environment. From the root of the repository, run:
    ```bash
    conda env create -f environment.yml
    conda activate ppicker
    ```

2. Install ProPicker itself:
    ```bash
    pip install .
    ```

3. You need the checkpoint of our pre-trained model, as well as the checkpoint of the TomoTwin model we used as prompt encoder.

    - You can download the ProPicker checkpoint here [here](https://drive.google.com/file/d/1mgzA576roiAA7WVPmkwsDXWiREQBkJvB/view?usp=sharing)

    - You can download the TomoTwin checkpoint by running `bash download_tomotwin_ckpt.sh`

    After downloading, place the files in the `ProPicker` directory. 

4. Set the following environment variables to point to the model files:
    ```bash
    export PROPICKER_MODEL_FILE=/abs/path/to/propicker.ckpt
    export TOMOTWIN_MODEL_FILE=/abs/path/to/tomotwin.pth
    ```

## Usage

### (I) Recommended: Python package and scripts 

We provide Jupyter notebooks as entry points for readers who want to try ProPicker or reproduce the qualitative behavior and performance trends reported in the paper.

- **Prompt-based picking:** [`python_tutorial/tutorial_1_prompt_based_picking/tutorial.ipynb`](https://github.com/MLI-lab/ProPicker/blob/main/python_tutorial/tutorial_1_prompt_based_picking/tutorial.ipynb)
- **Fine-tuning:** [`python_tutorial/tutorial_2_fine_tuning/tutorial.ipynb`](https://github.com/MLI-lab/ProPicker/blob/main/python_tutorial/tutorial_2_fine_tuning/tutorial.ipynb) (continues from prompt-based picking)

### (II) Experimental: Interactive exploration via GUI 

The GUI/CLI is for interactive exploration and application of ProPicker. The GUI uses the same underlying model and inference code, but is intended for exploratory use rather than for reproducing the quantitative results reported in the paper.

- **GUI tutorial:** see [`gui_tutorial/README.md`](https://github.com/MLI-lab/ProPicker/blob/main/gui_tutorial/README.md)



## Training ProPicker from Scratch

Training from scratch lives in `propicker/training_from_scratch/train.py`, with parameters in `propicker/training_from_scratch/train_cfg.py`.

To download the training data, you can use `datasets/download_train_data.sh`.

**Note:** The training data is large, so you might want to download it to a different location. To do this, modify `datasets/download_train_data.sh`; also adjust the training data path in `propicker/paths.py`.


## Note
This repository contains code copied and modified from [DeepETPicker](https://github.com/cbmi-group/DeepETPicker), [DeePiCt](https://github.com/ZauggGroup/DeePiCt), and [TomoTwin](https://github.com/MPI-Dortmund/tomotwin-cryoet). All derived code is explicitly marked.

## Citation
```
@article{wiedemann2026,
    title = {ProPicker: Promptable segmentation for particle picking in cryogenic electron tomography},
    journal = {Journal of Structural Biology},
    volume = {218},
    number = {2},
    pages = {108298},
    year = {2026},
    doi = {https://doi.org/10.1016/j.jsb.2026.108298},
    author = {Simon Wiedemann and Zalan Fabian and Mahdi Soltanolkotabi and Reinhard Heckel},
}
```
