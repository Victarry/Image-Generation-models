from pytorch_lightning import LightningModule
import torchmetrics
import torchvision
from src.utils.utils import get_logger
import torch
import matplotlib.pyplot as plt
import io
import PIL
from torchvision.transforms import ToTensor
import numpy as np
import torch.nn.functional as F

class BaseModel(LightningModule):
    def __init__(self, datamodule) -> None:
        super().__init__()
        self.console = get_logger()
        self.width = datamodule.width
        self.height = datamodule.height
        self.channels = datamodule.channels
        self.input_normalize = datamodule.transforms.normalize

    def adversarial_loss(self, y_hat, y):
        if self.hparams.loss_mode == "vanilla":
            return F.binary_cross_entropy_with_logits(y_hat, y)
        elif self.hparams.loss_mode == "lsgan":
            return F.mse_loss(y_hat, y)

    def get_grid_images(self, imgs, nimgs=64, nrow=8):
        imgs = imgs.reshape(
            -1, self.channels, self.height, self.width
        )
        if self.input_normalize:
            grid = torchvision.utils.make_grid(
                imgs[:nimgs], nrow=nrow, normalize=True, value_range=(-1, 1), pad_value=1
            )
        else:
            grid = torchvision.utils.make_grid(imgs[:nimgs], normalize=False, nrow=nrow, pad_value=1)
        return grid
    
    def log_hist(self, tensor, name):
        assert tensor.dim() == 1
        array = np.array(tensor.detach().cpu().numpy())
        self.logger.experiment.add_histogram(name, array, self.global_step)

    def log_images(self, imgs, name, nimgs=64, nrow=8):
        grid = self.get_grid_images(imgs, nimgs=nimgs, nrow=nrow)
        self.logger.experiment.add_image(name, grid, self.global_step)

    def image_float2int(self, imgs):
        if self.input_normalize:
            imgs = (imgs + 1) / 2
        imgs = (imgs * 255).to(torch.uint8)
        return imgs
    
    def tensor_to_array(self, *tensors):
        output = []
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                output.append(np.array(tensor.detach().cpu().numpy()))
            else:
                output.append(tensor)
        return output

    def plot_scatter(self, name, x, y, c=None, s=None, xlim=None, ylim=None):
        x, y, c, s = self.tensor_to_array(x, y, c, s)

        plt.figure()
        plt.scatter(x=x, y=y, s=s, c=c, cmap="tab10", alpha=1)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.title("Latent distribution")
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        plt.close()
        buf.seek(0)
        visual_image = ToTensor()(PIL.Image.open(buf))
        self.logger.experiment.add_image(name, visual_image, self.global_step)
    
    def on_validation_epoch_start(self) -> None:
        if self.hparams.eval_fid:
            self.fid = torchmetrics.FID().to(self.device)

    def validation_step(self, batch, batch_idx):
        if self.hparams.eval_fid:
            imgs, _ = batch
            self.fid.update(self.image_float2int(imgs), real=True)
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim).to(self.device)
            fake_imgs = self.forward(z)
            self.fid.update(self.image_float2int(fake_imgs), real=False)

    def on_validation_epoch_end(self):
        if self.hparams.eval_fid:
            self.log("metrics/fid", self.fid.compute())