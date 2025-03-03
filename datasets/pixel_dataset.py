import numpy as np
from torch.utils.data import Dataset
from typing import Any
from my_configs.config import SPRITES_NPY_PATH


class PixelDataset(Dataset):
    def __init__(self, transform: Any = None) -> None:
        self.data = np.load(SPRITES_NPY_PATH)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
