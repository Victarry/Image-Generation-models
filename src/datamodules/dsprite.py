import numpy as np
import pytorch_lightning as pl
from pathlib import Path
import torch
from .base import BaseDatamodule, get_transform
from .utils import url_retrive, CustomTensorDataset
from torch.utils.data import random_split


class DataModule(BaseDatamodule):
    def __init__(
        self,
        data_dir: str = "./data",
        width=64,
        height=64,
        channels=3,
        batch_size: int = 64,
        num_workers: int = 8,
        transforms=None,
    ):
        super().__init__(width, height, channels, batch_size, num_workers)
        self.data_dir = data_dir
        self.transform = get_transform(transforms)

        self.data_dir = Path(data_dir) / 'dsprite'
        self.data_file = self.data_dir / 'dsprites_64x64.npz'

    def prepare_data(self):
        # download
        URL = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
        if not self.data_file.exists():
            url_retrive(URL, self.data_file)

    def setup(self, stage=None):
        data = np.load(self.data_file, encoding='latin1')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        length = data.shape[0]
        full_data = CustomTensorDataset(data, transform=self.transform)
        self.train_data, self.val_data = random_split(full_data, [8*length // 10, 2*length // 10], generator=torch.Generator().manual_seed(666))
