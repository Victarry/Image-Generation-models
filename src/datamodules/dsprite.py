from genericpath import exists
from typing import Optional
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from .utils import url_retrive, CustomTensorDataset
from torchvision import transforms


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 8,
        normalize: bool = True,
        **kargs
    ):
        super().__init__()
        self.data_dir = Path(data_dir) / 'dsprite'
        self.data_file = self.data_dir / 'dsprites_64x64.npz'
        self.batch_size = batch_size
        self.num_workers = num_workers

        data_transform = []
        if normalize:
            data_transform.append(transforms.Normalize([0.5], [0.5]))

        self.transform = transforms.Compose(data_transform)
        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 64, 64)

    def prepare_data(self):
        # download
        URL = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
        if not self.data_file.exists():
            url_retrive(URL, self.data_file)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        data = np.load(self.data_file, encoding='latin1')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        self.train_data = CustomTensorDataset(data, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            multiprocessing_context="fork",
        )
