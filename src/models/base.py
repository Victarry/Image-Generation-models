from pytorch_lightning import LightningModule
import torchvision
from src.utils.utils import get_logger
import torch
import matplotlib.pyplot as plt
import io
import PIL
from torchvision.transforms import ToTensor


class BaseModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.console = get_logger()

    def get_grid_images(self, imgs):
        imgs = imgs.reshape(
            -1, self.hparams.channels, self.hparams.height, self.hparams.width
        )
        if self.hparams.input_normalize:
            grid = torchvision.utils.make_grid(
                imgs[:64], normalize=True, value_range=(-1, 1)
            )
        else:
            grid = torchvision.utils.make_grid(imgs[:64], normalize=False)
        return grid

    def log_images(self, imgs, name):
        grid = self.get_grid_images(imgs)
        self.logger.experiment.add_image(name, grid, self.global_step)

    def image_float2int(self, imgs):
        if self.hparams.input_normalize:
            imgs = (imgs + 1) / 2
        imgs = (imgs * 255).to(torch.uint8)
        return imgs

    def plot_scatter(self, name, x, y, c=None, s=None, xlim=None, ylim=None):
        plt.figure()
        plt.scatter(x=x, y=y, s=s, c=c, cmap="tab10", alpha=1)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.title("Latent distribution")
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        visual_image = ToTensor()(PIL.Image.open(buf))
        self.logger.experiment.add_image(name, visual_image, self.global_step)