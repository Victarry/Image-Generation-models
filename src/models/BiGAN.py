from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from src.utils import utils
from .base import BaseModel
import itertools
import torchmetrics
from src.networks.basic import MLPEncoder


class Discriminator(nn.Module):
    def __init__(self, encoder, input_channel, latent_dim, hidden_dim) -> None:
        super().__init__()
        self.dis_z = MLPEncoder(
            input_channel=latent_dim,
            output_channel=hidden_dim,
            width=1,
            height=1,
            hidden_dims=[hidden_dim, hidden_dim],
            first_batch_norm=True,
            last_batch_norm=True,
            output_act="leaky_relu"
        )
        self.dis_x = hydra.utils.instantiate(
            encoder, input_channel=input_channel, output_channel=hidden_dim
        )
        self.dis_pair = MLPEncoder(
            input_channel=2 * hidden_dim,
            output_channel=1,
            width=1,
            height=1,
            hidden_dims=[hidden_dim],
            first_batch_norm=True
        )

    def forward(self, x, z):
        z_feature = self.dis_z(z)
        x_feature = self.dis_x(x)
        # z_feature = torch.zeros_like(x_feature).type_as(x_feature)
        concat_feature = torch.cat((z_feature, x_feature), dim=1)
        return self.dis_pair(concat_feature)


class BiGAN(BaseModel):
    def __init__(
        self,
        channels,
        width,
        height,
        encoder,
        decoder,
        dis_x,
        latent_dim=100,
        hidden_dim=512,
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
        self.decoder = hydra.utils.instantiate(
            decoder, input_channel=latent_dim, output_channel=channels
        )
        self.encoder = hydra.utils.instantiate(
            encoder, input_channel=channels, output_channel=latent_dim
        )
        self.discriminator = Discriminator(encoder, channels, latent_dim, hidden_dim)

    def forward(self, z):
        output = self.decoder(z)
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
            fake_imgs = self.decoder(z)

            encode_z = self.encoder(imgs)

            # log sampled images
            self.log_images(fake_imgs, "generated_images")

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1).to(self.device)
            fake = torch.zeros(imgs.size(0), 1).to(self.device)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(
                self.discriminator(fake_imgs, z), valid
            ) + self.adversarial_loss(self.discriminator(imgs, encode_z), fake)
            self.log("train_loss/g_loss", g_loss)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # real loss
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            real_z = self.encoder(imgs)
            real_logit = self.discriminator(imgs, real_z)
            real_loss = self.adversarial_loss(real_logit, valid)

            # fake loss
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)
            fake_imgs = self.decoder(z)
            fake_logit = self.discriminator(fake_imgs, z)
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
        imgs = self.decoder(z)
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

        g_param = itertools.chain(self.encoder.parameters(), self.decoder.parameters())
        d_param = self.discriminator.parameters()
        if self.hparams.optim == "adam":
            opt_g = torch.optim.Adam(g_param, lr=lrG, betas=(b1, b2))
            opt_d = torch.optim.Adam(d_param, lr=lrD, betas=(b1, b2))
        elif self.hparams.optim == "sgd":
            opt_g = torch.optim.SGD(g_param, lr=lrG)
            opt_d = torch.optim.SGD(d_param, lr=lrD)
        return [opt_g, opt_d]
