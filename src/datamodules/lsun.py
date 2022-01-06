from typing import Optional
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import LSUN
from torchvision import transforms
from urllib.request import Request, urlopen
from os.path import join
import subprocess
from pathlib import Path

"""
Download of LSUN dataset refer to https://github.com/fyu/lsun
Place dataset file to data_dir/lsun
"""


class LSUNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 8,
        width=64,
        height=64,
        categories=["bedroom"],
        **kargs
    ):
        super().__init__()
        self.data_dir = data_dir + "/lsun"
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [transforms.Resize((width, height)), transforms.ToTensor()]
        )
        self.categories = categories

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, 64, 64)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_data = LSUN(
            self.data_dir,
            classes=[x + "_train" for x in self.categories],
            transform=self.transform,
        )
        self.test_data = LSUN(
            self.data_dir,
            classes=[x + "_test" for x in self.categories],
            transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    data = LSUNDataModule()
    data.prepare_data()
    data.setup()
    data.train_dataloader()
