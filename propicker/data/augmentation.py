#%%
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import scipy.ndimage
from numpy.typing import NDArray
import torch

class Augmentation(ABC):
    def __init__(self, prob: float = 1.0, seed=None):
        self.prob = prob
        self.seed = seed
        self.running_seed = seed
    
    @abstractmethod
    def volume_forward(self, volume):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __call__(self, item):
        if not isinstance(item, dict):
            raise ValueError("Item must be a dictionary")
        if not "model_input" in item:
            raise ValueError("Item must contain key 'model_input'")
        if not "model_targets" in item:
            raise ValueError("Item must contain key 'model_targets'")
        
        # spatial augmentations should also be applied to the targets
        if self._is_spatial:
            # concat model_input and model_targets to 4D tensor to ensure same augmentation is applied to both
            assert len(item["model_input"].shape) == 3
            assert len(item["model_targets"].shape) == 4
            model_input = item["model_input"].unsqueeze(0)
            helper_vol = torch.cat([model_input, item["model_targets"]], dim=0)
            helper_vol = self.volume_forward(helper_vol)
            item["model_input"] = helper_vol[0]
            item["model_targets"] = helper_vol[1:].round()
        else:
            item["model_input"] = self.volume_forward(item["model_input"])
        if not torch.is_tensor(item["model_input"]):
            item["model_input"] = torch.tensor(item["model_input"].copy())
        if not torch.is_tensor(item["model_targets"]):
            item["model_targets"] = torch.tensor(item["model_targets"].copy())
        return item

class AugmentationPipeline:
    def __init__(self, augs: list[Augmentation], seed=None):
        self.augs = augs
        self.seed = seed
        self.running_seed = seed

    def __str__(self):
        for aug in self.augs:
            return str(aug)

    def __call__(self, item):
        for aug in self.augs:  
            if np.random.RandomState(self.running_seed).rand() < aug.prob:
                item = aug(item)          
        if self.seed is not None:
            self.running_seed += 1
        return item

class VoxelDropout(Augmentation):
    '''
    Replaces a random amount of voxels with the volume mean
    '''

    def __init__(self, ratio : list[float,float], prob=1.0, seed=None):
        super().__init__(prob=prob)
        self.ratio = ratio
        self._is_spatial = False
        self.seed = seed
        self.running_seed = seed

    def volume_forward(self, volume):
        if isinstance(self.ratio, float):
            rand_ratio = self.ratio
        else:
            rand_ratio = self.ratio[0] + np.random.RandomState(self.running_seed).rand() * (self.ratio[1] - self.ratio[0])
        mean_val = 0 #np.mean(volume)
        drop = np.random.RandomState(self.running_seed).binomial(
            n=1, p=1 - rand_ratio, size=(volume.shape[0], volume.shape[1],volume.shape[2])
        )
        if len(volume.shape) == 3:
            volume[drop == 0] = mean_val
        elif len(volume.shape) == 4:
            for k in range(volume.shape[0]):
                volume[k][drop[k] == 0] = mean_val
        if self.seed is not None:
            self.running_seed += 1
        return volume

    def __str__(self):
        return f"Voxeldropout (Ratio {self.ratio}); Probability: {self.prob}"

class AddNoise(Augmentation):
    '''
    Sets blocks of random size to 0
    '''
    def __init__(self, sigma : list[float,float], prob=1.0, seed=None):
        super().__init__(prob=prob)
        self.sigma = sigma
        self._is_spatial = False
        self.seed = seed
        self.running_seed = seed

    def volume_forward(self, volume):
        noise = (np.random.RandomState(self.running_seed).randn(*volume.shape) * (np.random.RandomState(self.running_seed).uniform(*self.sigma))).astype("f")
        if len(volume.shape) == 3:
            noisy_volume = np.add(volume, noise)
        elif len(volume.shape) == 4:
            noisy_volume = np.zeros_like(volume)
            for k in range(volume.shape[0]):
                noisy_volume[k] = np.add(volume[k], noise[k])
        else:
            raise ValueError("Volume must be 3D or 4D")
        if self.seed is not None:
            self.running_seed += 1
        return noisy_volume

    def __str__(self):
        return f"AddNoise (Sigma: {self.sigma}); Probability: {self.prob}"

class RotateFull(Augmentation):
    '''
    Does custrom rotations around the specified axis
    '''

    def __init__(self, axes=(0, 1), prob=1.0, seed=None):
        super().__init__(prob=prob)
        self.axes = axes
        self._is_spatial = True
        self.seed = seed
        self.running_seed = seed

    def volume_forward(self, volume):
        angle = np.random.RandomState(self.running_seed).rand()*360
        if len(volume.shape) == 3:
            rotated = scipy.ndimage.rotate(volume, angle=angle, axes=self.axes, reshape=False, mode='reflect')
        elif len(volume.shape) == 4:
            rotated = np.zeros_like(volume)
            for k in range(volume.shape[0]):
                rotated[k] = scipy.ndimage.rotate(volume[k], angle=angle, axes=self.axes, reshape=False, mode='reflect')
        else:
            raise ValueError("Volume must be 3D or 4D")
        if self.seed is not None:
            self.running_seed += 1
        # round rotated volume to ensure binary values
        rotated = np.round(rotated).astype("float32")
        return rotated

    def __str__(self):
        return f"RotateFull (Axes: {self.axes}); Probability: {self.prob}"


class Flip(Augmentation):
    '''
    Flips the image along the specified axis
    '''

    def __init__(self, axis=(0,1), prob=1.0, seed=None):
        super().__init__(prob=prob)
        self.axis = axis
        self._is_spatial = True
        self.seed = seed
        self.running_seed = seed

    def volume_forward(self, volume):
        # ensure volume is a numpy array
        if torch.is_tensor(volume):
            volume = volume.numpy()
        if len(volume.shape) == 3:
            flipped = np.flip(volume, axis=self.axis)
        elif len(volume.shape) == 4:
            flipped = np.zeros_like(volume)
            for k in range(volume.shape[0]):
                flipped[k] = np.flip(volume[k], axis=self.axis)
        else:
            raise ValueError("Volume must be 3D or 4D")
        return flipped
        
    def __str__(self):
        return f"Flip (Axis: {self.axis}); Probability: {self.prob}"