from typing import Optional
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms


class CelebADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 8,
        width=64,
        height=64,
        **kargs
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [transforms.Resize((width, height)), transforms.ToTensor()]
        )

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, 64, 64)

    def prepare_data(self):
        # download
        CelebA(self.data_dir, split="all", download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_data = CelebA(
            self.data_dir, split="train", transform=self.transform
        )
        self.test_data = CelebA(
            self.data_dir, split="test", transform=self.transform
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
