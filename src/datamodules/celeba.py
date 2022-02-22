from torchvision.datasets import CelebA
from .base import get_transform, BaseDatamodule

class CelebADataModule(BaseDatamodule):
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

    def prepare_data(self):
        CelebA(self.data_dir, split="all", download=True)

    def setup(self, stage=None):
        self.train_data = CelebA(self.data_dir, split="train", transform=self.transform)
        self.val_data = CelebA(self.data_dir, split="test", transform=self.transform)
