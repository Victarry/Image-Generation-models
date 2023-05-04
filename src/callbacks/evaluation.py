import pytorch_lightning as pl
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
from src.models.base import ValidationResult


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
            self.fid = FrechetInceptionDistance().to(pl_module.device)

    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch, batch_idx):
        if pl_module.channels == 3 and trainer.current_epoch % self.every_n_epoch == 0:
            real_imgs, fake_images = outputs.real_image, outputs.fake_image
            self.fid.update(self.image_float2int(real_imgs, pl_module), real=True)
            self.fid.update(self.image_float2int(fake_images, pl_module), real=False)

    def on_validation_epoch_end(self, trainer, pl_module: pl.LightningModule):
        if pl_module.channels == 3 and trainer.current_epoch % self.every_n_epoch == 0:
            pl_module.log("metrics/fid", self.fid.compute(), on_epoch=True)
