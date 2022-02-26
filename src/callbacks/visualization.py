import numpy as np
import pytorch_lightning as pl
from pathlib import Path
import torchvision
import torch
from src.models.base import ValidationResult
import torch.distributions as D

def get_grid_images(imgs, model, nimgs=64, nrow=8):
    if model.input_normalize:
        grid = torchvision.utils.make_grid(
            imgs[:nimgs], nrow=nrow, normalize=True, value_range=(-1, 1), pad_value=1
        )
    else:
        grid = torchvision.utils.make_grid(imgs[:nimgs], normalize=False, nrow=nrow, pad_value=1)
    return grid

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

            fake_grid = get_grid_images(outputs.fake_image, pl_module)
            trainer.logger.experiment.add_image("images/sample", fake_grid, global_step=trainer.current_epoch)
            torchvision.utils.save_image(fake_grid, result_path / f"{trainer.current_epoch}.jpg")


class TraverseLatentCallback(pl.Callback):
    def __init__(self, col=10, row=10) -> None:
        super().__init__()
        self.col = col
        self.row = row
    
    def generate_traverse_images(self, pl_module):
        row, col = min(10, pl_module.hparams.latent_dim), 11
        fixed_z = torch.randn(1, 1, pl_module.hparams.latent_dim).repeat(row, col, 1).reshape(row, col, -1).to(pl_module.device)
        variation_z = torch.linspace(-3, 3, col).to(pl_module.device)
        for i in range(row):
            fixed_z[i, :, i] += variation_z # i-th row correspondes to i-th latent unit variation
        imgs = pl_module.forward(fixed_z.reshape(row*col, -1))
        grid = get_grid_images(imgs, pl_module, nimgs=row*col, nrow=col)
        return grid
    
    def on_validation_epoch_end(self, trainer, pl_module):
        grid1 = self.generate_traverse_images(pl_module)
        grid2 = self.generate_traverse_images(pl_module)
        trainer.logger.experiment.add_image("sample/traverse_latents_1", grid1, global_step=trainer.current_epoch)
        trainer.logger.experiment.add_image("sample/traverse_latents_2", grid2, global_step=trainer.current_epoch)

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
            self.plot_scatter("val/latent distributions", x=latents_array[:, 0][sort_idx], y=latents_array[:,1][sort_idx], 
                c=labels_array[sort_idx], xlim=(-3, 3), ylim=(-3, 3))