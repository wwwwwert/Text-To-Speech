from typing import Callable, List

from torch import Tensor

from hw_tts.augmentations.base import AugmentationBase


class SequentialAugmentation(AugmentationBase):
    def __init__(self, augmentation_list: List[Callable]):
        self.augmentation_list = augmentation_list

    def __call__(self, data: Tensor) -> Tensor:
        x = data
        for augmentation in self.augmentation_list:
            x = augmentation(x)
        return x
