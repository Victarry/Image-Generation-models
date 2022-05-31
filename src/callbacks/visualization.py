from random import randint
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
import torchvision
import matplotlib.pyplot as plt
import io
from torchvision.transforms import ToTensor
import torch
from src.models.base import ValidationResult
import PIL

class SampleImagesCallback(pl.Callback):
    def __init__(self, batch_size=64, every_n_epochs=1):
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs

    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch, batch_idx, dataloader_idx):
        if trainer.current_epoch % self.every_n_epochs == 0 and batch_idx == 0:
            result_path = Path("results")
            result_path.mkdir(parents=True, exist_ok=True)

            real_grid = get_grid_images(outputs.real_image, pl_module)
            trainer.logger.experiment.add_image("images/real", real_grid, global_step=trainer.current_epoch)

            if outputs.recon_image is not None:
                recon_grid = get_grid_images(outputs.recon_image, pl_module)
                trainer.logger.experiment.add_image("images/recon", recon_grid, global_step=trainer.current_epoch)

            if outputs.fake_image is not None:
                fake_grid = get_grid_images(outputs.fake_image, pl_module)
                trainer.logger.experiment.add_image("images/sample", fake_grid, global_step=trainer.current_epoch)
                torchvision.utils.save_image(fake_grid, result_path / f"{trainer.current_epoch}.jpg")

            for key in outputs.others:
                grid = get_grid_images(outputs.others[key], pl_module)
                trainer.logger.experiment.add_image(f"images/{key}", grid, global_step=trainer.current_epoch)


class TraverseLatentCallback(pl.Callback):
    def __init__(self, col=10, row=10) -> None:
        super().__init__()
        self.col = col
        self.row = row
    
    def generate_traverse_images(self, pl_module, fixed_z=None):
        row, col = 11, min(10, pl_module.hparams.latent_dim)
        if fixed_z is None:
            fixed_z = torch.randn(1, 1, pl_module.hparams.latent_dim).repeat(row, col, 1).reshape(row, col, -1).to(pl_module.device)
        else:
            fixed_z = fixed_z.reshape(1, 1, pl_module.hparams.latent_dim).repeat(row, col, 1).reshape(row, col, -1)
        variation_z = torch.linspace(-3, 3, row).to(pl_module.device)
        for i in range(col):
            fixed_z[:, i, i] = variation_z # i-th column correspondes to i-th latent unit variation
        imgs = pl_module.forward(fixed_z.reshape(row*col, -1))
        grid = get_grid_images(imgs, pl_module, nimgs=row*col, nrow=col)
        return grid
    
    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch, batch_idx, dataloader_idx: int):
        if batch_idx == 0:
            self.z = outputs.encode_latent
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.z is not None:
            grid1 = self.generate_traverse_images(pl_module, self.z[3])
            trainer.logger.experiment.add_image("sample/fixed_traverse_latents_1", grid1, global_step=trainer.current_epoch)
        if self.z is not None:
            grid2 = self.generate_traverse_images(pl_module, self.z[6])
            trainer.logger.experiment.add_image("sample/fixed_traverse_latents_2", grid2, global_step=trainer.current_epoch)

        grid = self.generate_traverse_images(pl_module)
        trainer.logger.experiment.add_image("sample/random_traverse_latents", grid, global_step=trainer.current_epoch)

class Visual2DSpaecCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if pl_module.hparams.latent_dim == 2:
            x = torch.tensor(np.linspace(-3, 3, 20)).to(pl_module.device)
            y = torch.tensor(np.linspace(3, -3, 20)).to(pl_module.device)
            xx, yy = torch.meshgrid([y, x])
            latent = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1) # (20*20, 2)
            imgs = pl_module.forward(latent)
            grid_imgs = get_grid_images(imgs, pl_module, nimgs=400, nrow=20)
            trainer.logger.experiment.add_image("sample/grid_imgs", grid_imgs, global_step=trainer.current_epoch)

class LatentVisualizationCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
    
    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if pl_module.hparams.latent_dim == 2:
            self.latents = []
            self.labels = []
    
    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch, batch_idx, dataloader_idx):
        if pl_module.hparams.latent_dim == 2:
            self.latents.append(outputs.encode_latent)
            self.labels.append(outputs.label)

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.hparams.latent_dim == 2:
            latents_array = torch.cat(self.latents).cpu().numpy()
            labels_array = torch.cat(self.labels).cpu().numpy()
            sort_idx = np.argsort(labels_array)
            self.latents = []
            self.labels = []
            img = make_scatter(x=latents_array[:, 0][sort_idx], y=latents_array[:,1][sort_idx], 
                c=labels_array[sort_idx], xlim=(-3, 3), ylim=(-3, 3))
            trainer.logger.experiment.add_image("val/latent distributions", img, global_step=trainer.current_epoch)


def tensor_to_array(*tensors):
    output = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            output.append(np.array(tensor.detach().cpu().numpy()))
        else:
            output.append(tensor)
    return output

def make_scatter(x, y, c=None, s=None, xlim=None, ylim=None):
    x, y, c, s = tensor_to_array(x, y, c, s)

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
    return visual_image

def get_grid_images(imgs, model, nimgs=64, nrow=8):
    if model.input_normalize:
        grid = torchvision.utils.make_grid(
            imgs[:nimgs], nrow=nrow, normalize=True, value_range=(-1, 1), pad_value=1
        )
    else:
        grid = torchvision.utils.make_grid(imgs[:nimgs], normalize=False, nrow=nrow, pad_value=1)
    return grid