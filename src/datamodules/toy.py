from typing import Optional
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


class MixtureGaussianDataset(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def train_dataloader(self):
        return super().train_dataloader()
