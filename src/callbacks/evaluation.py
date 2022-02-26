from glob import glob
import pytorch_lightning as pl
from pathlib import Path
import torchmetrics
import torchvision
import torch
from src.models.base import ValidationResult


def get_grid_images(imgs, model, nimgs=64, nrow=8):
    if model.input_normalize:
        grid = torchvision.utils.make_grid(
            imgs[:nimgs], nrow=nrow, normalize=True, value_range=(-1, 1), pad_value=1
        )
    else:
        grid = torchvision.utils.make_grid(imgs[:nimgs], normalize=False, nrow=nrow, pad_value=1)
    return grid

class FIDEvaluationCallback(pl.Callback):
    def __init__(self, every_n_epochs=1):
        self.every_n_epoch = every_n_epochs

    def image_float2int(self, imgs, pl_module):
        if pl_module.input_normalize:
            imgs = (imgs + 1) / 2
        imgs = (imgs * 255).to(torch.uint8)
        return imgs

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if pl_module.channels == 3 and trainer.current_epoch % self.every_n_epoch == 0:
            self.fid = torchmetrics.FID().to(pl_module.device)

    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch, batch_idx, dataloader_idx):
        if pl_module.channels == 3 and trainer.current_epoch % self.every_n_epoch == 0:
            real_imgs, fake_images = outputs.real_image, outputs.fake_image
            self.fid.update(self.image_float2int(real_imgs, pl_module), real=True)
            self.fid.update(self.image_float2int(fake_images, pl_module), real=False)

    def on_validation_epoch_end(self, trainer, pl_module: pl.LightningModule):
        if pl_module.channels == 3 and trainer.current_epoch % self.every_n_epoch == 0:
            pl_module.log("metrics/fid", self.fid.compute())

class SampleImagesCallback(pl.Callback):
    def __init__(self, batch_size=64, every_n_epochs=1):
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs

    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch, batch_idx, dataloader_idx):
        if trainer.current_epoch % self.every_n_epochs == 0 and batch_idx == 0:
            result_path = Path("results")
            result_path.mkdir(parents=True, exist_ok=True)
            fake_grid = get_grid_images(outputs.fake_image, pl_module)
            real_grid = get_grid_images(outputs.real_image, pl_module)
            trainer.logger.experiment.add_image("images/real", real_grid, global_step=trainer.current_epoch)
            trainer.logger.experiment.add_image("images/sample", fake_grid, global_step=trainer.current_epoch)
            torchvision.utils.save_image(fake_grid, result_path / f"{trainer.current_epoch}.jpg")

class TraverseLatentCallback(pl.Callback):
    def __init__(self, col=10, row=10) -> None:
        super().__init__()
        self.col = col
        self.row = row
    
    def generate_traverse_images(self, pl_module):
        row, col = 10, 11
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