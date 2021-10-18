"""
Traditional Unconditional GANs, with different loss modes including:
1. Binary Cross Entroy (vanilla_gan)
2. Least Square Error(lsgan)
"""
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from src.utils import utils
from .base import BaseModel
import torchmetrics


class GAN(BaseModel):
    def __init__(
        self,
        channels,
        width,
        height,
        netG,
        netD,
        latent_dim=100,
        loss_mode="vanilla",
        lrG: float = 0.0002,
        lrD: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        input_normalize=True,
        optim="adam",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.generator = hydra.utils.instantiate(
            netG, input_channel=latent_dim, output_channel=channels
        )
        self.discriminator = hydra.utils.instantiate(
            netD, input_channel=channels, output_channel=1
        )

    def forward(self, z):
        output = self.generator(z)
        output = output.reshape(
            z.shape[0], self.hparams.channels, self.hparams.height, self.hparams.width
        )
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch  # (N, C, H, W)
        if self.hparams.input_normalize:
            imgs = imgs * 2 - 1

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)  # (N, latent_dim)
        z = z.type_as(imgs)

        # train generator, pytorch_lightning will automatically set discriminator requires_gard as False
        if optimizer_idx == 0:

            # generate images
            generated_imgs = self(z)

            # log sampled images
            self.log_images(generated_imgs, "generated_images")

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(generated_imgs), valid)
            self.log("train_loss/g_loss", g_loss, prog_bar=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # real loss
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            real_logit = self.discriminator(imgs)
            real_loss = self.adversarial_loss(real_logit, valid)

            # fake loss
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)
            fake_logit = self.discriminator(self(z).detach())
            fake_loss = self.adversarial_loss(fake_logit, fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("train_loss/d_loss", d_loss)
            self.log("train_log/real_logit", real_logit.mean())
            self.log("train_log/fake_logit", fake_logit.mean())

            return d_loss

    def on_train_epoch_end(self):
        result_path = Path("results")
        result_path.mkdir(parents=True, exist_ok=True)
        if hasattr(self, "z"):
            z = self.z
        else:
            self.z = z = torch.randn(64, self.hparams.latent_dim).to(self.device)
        imgs = self.generator(z)
        grid = self.get_grid_images(imgs)
        torchvision.utils.save_image(grid, result_path / f"{self.current_epoch}.jpg")

    def on_validation_epoch_start(self) -> None:
        self.fid = torchmetrics.FID().to(self.device)

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        self.fid.update(self.image_float2int(imgs), real=True)
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim).to(self.device)
        fake_imgs = self(z)
        self.fid.update(self.image_float2int(fake_imgs), real=False)

    def on_validation_epoch_end(self):
        self.log("metrics/fid", self.fid.compute())

    def adversarial_loss(self, y_hat, y):
        if self.hparams.loss_mode == "vanilla":
            return F.binary_cross_entropy_with_logits(y_hat, y)
        elif self.hparams.loss_mode == "lsgan":
            return F.mse_loss(y_hat, y)

    def configure_optimizers(self):
        lrG = self.hparams.lrG
        lrD = self.hparams.lrD
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        if self.hparams.optim == "adam":
            opt_g = torch.optim.Adam(
                self.generator.parameters(), lr=lrG, betas=(b1, b2)
            )
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(), lr=lrD, betas=(b1, b2)
            )
        elif self.hparams.optim == "sgd":
            opt_g = torch.optim.SGD(self.generator.parameters(), lr=lrG)
            opt_d = torch.optim.SGD(self.discriminator.parameters(), lr=lrD)
        return [opt_g, opt_d]
