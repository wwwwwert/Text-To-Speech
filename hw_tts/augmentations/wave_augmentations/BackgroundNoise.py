from os import listdir
from os.path import isfile, join

from hw_tts.augmentations.base import AugmentationBase
from torch import Tensor
from torch_audiomentations import AddBackgroundNoise


class BackgroundNoise(AugmentationBase):
    def __init__(self, sample_rate:int, *args, **kwargs) -> None:
        super().__init__()
        dataset = "hw_tts/augmentations/wave_augmentations/background_noise"
        wavs_paths = [join(dataset, f) for f in listdir(dataset) if isfile(join(dataset, f))]
        self._aug = AddBackgroundNoise(wavs_paths, p=0.2, sample_rate=sample_rate)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
