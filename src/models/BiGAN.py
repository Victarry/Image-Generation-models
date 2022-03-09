import itertools

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
from src.networks.basic import MLPEncoder
from src.utils.losses import adversarial_loss
from torch import nn

from .base import BaseModel, ValidationResult


class BiGAN(BaseModel):
    def __init__(
        self,
        datamodule,
        encoder,
        decoder,
        latent_dim=100,
        hidden_dim=512,
        loss_mode="vanilla",
        lrG: float = 0.0002,
        lrD: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()

        # networks
        self.decoder = hydra.utils.instantiate(
            decoder, input_channel=latent_dim, output_channel=self.channels
        )
        self.encoder = hydra.utils.instantiate(
            encoder, input_channel=self.channels, output_channel=latent_dim
        )
        self.discriminator = Discriminator(encoder, self.channels, latent_dim, hidden_dim)
        self.automatic_optimization = False

    def forward(self, z):
        output = self.decoder(z)
        output = output.reshape(
            z.shape[0], self.channels, self.height, self.width
        )
        return output

    def configure_optimizers(self):
        lrG = self.hparams.lrG
        lrD = self.hparams.lrD
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        g_param = itertools.chain(self.encoder.parameters(), self.decoder.parameters())
        d_param = self.discriminator.parameters()
        opt_g = torch.optim.Adam(g_param, lr=lrG, betas=(b1, b2))
        opt_d = torch.optim.Adam(d_param, lr=lrD, betas=(b1, b2))
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):
        imgs, _ = batch  # (N, C, H, W)
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim).to(self.device)  # (N, latent_dim)

        optim_g, optim_d = self.optimizers()

        real_pair = imgs, self.encoder(imgs)
        fake_pair = self.decoder(z), z

        real_logit = self.discriminator(*real_pair)
        fake_logit = self.discriminator(*fake_pair)

        mode = self.hparams.loss_mode
        g_loss = adversarial_loss(real_logit, False, mode) + adversarial_loss(fake_logit, True, mode)
        d_loss = adversarial_loss(real_logit, True, mode) + adversarial_loss(fake_logit, False, mode)

        optim_g.zero_grad()
        self.manual_backward(g_loss, retain_graph=True)
        optim_g.step()

        optim_d.zero_grad()
        self.manual_backward(d_loss, inputs=list(self.discriminator.parameters()), retain_graph=True)
        optim_d.step()

        self.log("train_loss/g_loss", g_loss)
        self.log("train_loss/d_loss", d_loss)
        self.log("train_log/real_logit", real_logit.mean())
        self.log("train_log/fake_logit", fake_logit.mean())

    def validation_step(self, batch, batch_idx):
        img, _ = batch
        z = torch.randn(img.shape[0], self.hparams.latent_dim).to(self.device)
        fake_img = self.forward(z)

        encode_z = self.encoder(img)
        recon_img = self.decoder(encode_z)
        return ValidationResult(real_image=img, fake_image=fake_img, recon_image=recon_img, encode_latent=encode_z)


class Discriminator(nn.Module):
    def __init__(self, encoder, input_channel, latent_dim, hidden_dim) -> None:
        super().__init__()
        self.dis_z = MLPEncoder(
            input_channel=latent_dim,
            output_channel=hidden_dim,
            width=1,
            height=1,
            hidden_dims=[hidden_dim, hidden_dim],
            output_act="leaky_relu",
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
        )

    def forward(self, x, z):
        z_feature = self.dis_z(z)
        x_feature = self.dis_x(x)
        concat_feature = torch.cat((z_feature, x_feature), dim=1)
        return self.dis_pair(concat_feature)
