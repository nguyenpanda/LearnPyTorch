from pathlib import Path
from typing import Iterable, Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class AutoencoderCSVDataset(Dataset):

    def __init__(self, csv_path: Path,
                 transform: Callable[[Union[np.ndarray, Image]], torch.Tensor],
                 xy_range: Iterable[int] = (5, 15),
                 wh_range: Iterable[int] = (7, 15)):
        super().__init__()
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.path = csv_path
        self.xy_range = xy_range
        self.wh_range = wh_range

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        data_row = self.df.iloc[item]
        y = data_row.iloc[1:].to_numpy().reshape((28, 28)).astype(np.uint8)

        x = y.copy()

        xy = np.random.randint(*self.xy_range, size=(2,))
        wh = np.random.randint(*self.wh_range, size=(2,))

        valid_range = np.minimum(xy + wh, 27)
        x[xy[0]:valid_range[0], xy[1]:valid_range[1]] = 0

        return self.transform(x), self.transform(y)


class AutoencoderDirValidationDataset(Dataset):

    def __init__(self, path: Path,
                 transform: Callable[[Union[np.ndarray, Image]], torch.Tensor]):
        self.path = path
        self.list_img_path = list(path.glob('*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.list_img_path)

    def __getitem__(self, item) -> torch.Tensor:
        img_path = self.list_img_path[item]
        img = self.get_img(img_path)
        return self.transform(img)

    @staticmethod
    def get_img(img_path):
        img = plt.imread(img_path)
        if img.ndim > 2:
            img = img[..., 0]
        return img
