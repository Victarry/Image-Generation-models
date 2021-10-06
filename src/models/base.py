from pytorch_lightning import LightningModule
import torchvision
from src.utils.utils import get_logger


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
