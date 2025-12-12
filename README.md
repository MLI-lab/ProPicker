# ProPicker

## Update
We implemented a CLI/GUI workflow for prompt-based picking with ProPicker. See [here](gui_tutorial/README.md) for details.

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

    - You can download the ProPicker checkpoint here [here](https://drive.google.com/file/d/1lFDIJdAc99QVfGDuhOu89vEXYlz-oHQh/view?usp=share_link)

    - You can download the TomoTwin checkpoint by running `bash download_tomotwin_ckpt.sh`

    After downloading, place the files in the `ProPicker` directory. 

4. Set the following environment variables to point to the model files:
    ```bash
    export PROPICKER_MODEL_FILE=/abs/path/to/propicker.ckpt
    export TOMOTWIN_MODEL_FILE=/abs/path/to/tomotwin.pth
    ```

## Usage
There are two main workflows: prompt-based picking and fine-tuning:

### Workflow 1: Prompt-Based Picking 

Prompt-based picking can be done via both the Python package and the CLI/GUI.
We provide tutorials for both:

- **Python Package Tutorial:** See the `TUTORIAL1:empiar10988_prompt_based_picking.ipynb` notebook.

- **GUI/CLI Tutorial:** See the our [quickstart](gui_tutorial/README.md).


### Workflow 2: Fine-Tuning ProPicker

Fine-tuning currently only works via the Python package and additional scripts. See the `TUTORIAL2:empiar10988_fine_tuning.ipynb` notebook, which continues from the first tutorial.


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
