import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import LSUN

from datamodules.base import get_transform

"""
Download of LSUN dataset refer to https://github.com/fyu/lsun
Place dataset file to data_dir/lsun
"""

class LSUNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        width=64,
        height=64,
        channels=3,
        batch_size: int = 64,
        num_workers: int = 8,
        transforms=None,
        categories=["bedroom"]
    ):
        super().__init__(width, height, channels, batch_size, num_workers)
        self.data_dir = data_dir + "/lsun"
        self.transform = get_transform(transforms)
        self.categories = categories

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

if __name__ == "__main__":
    data = LSUNDataModule()
    data.prepare_data()
    data.setup()
    data.train_dataloader()