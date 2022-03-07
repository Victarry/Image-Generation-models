from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from src.utils import utils
from .base import BaseModel, ValidationResult


class WGAN(BaseModel):
    def __init__(
        self,
        datamodule,
        netG,
        netD,
        latent_dim=100,
        n_critic=5,
        lrG: float = 2e-4,
        lrD: float = 2e-4,
        b1: float = 0,
        b2: float = 0.9,
        gp_weight=10,
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()

        self.generator = hydra.utils.instantiate(netG, input_channel=latent_dim, output_channel=self.channels, norm_type="instance")
        self.discriminator = hydra.utils.instantiate(netD, input_channel=self.channels, output_channel=1, norm_type="instance")

    def forward(self, z):
        output = self.generator(z)
        output = output.reshape(
            z.shape[0], self.channels, self.height, self.width
        )
        return output

    def configure_optimizers(self):
        lrG = self.hparams.lrG
        lrD = self.hparams.lrD
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lrG, betas=(b1, b2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lrD, betas=(b1, b2)
        )
        return [
            {"optimizer": opt_g, "frequency": 1},
            {"optimizer": opt_d, "frequency": self.hparams.n_critic},
        ]

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch  # (N, C, H, W)

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)  # (N, latent_dim)
        z = z.type_as(imgs)

        if optimizer_idx == 0:
            generated_imgs = self(z)

            g_loss = -torch.mean(self.discriminator(generated_imgs))
            self.log("train_loss/g_loss", g_loss, prog_bar=True)
            return g_loss

        if optimizer_idx == 1:
            # real loss
            real_loss = -self.discriminator(imgs).mean()

            # fake loss
            fake_imgs = self(z)
            fake_loss = self.discriminator(fake_imgs.detach()).mean()

            # gradient panelty
            N = imgs.shape[0]
            lerp = torch.zeros(N, 1, 1, 1).uniform_().to(self.device)
            inter_x = torch.tensor(
                lerp * imgs + (1 - lerp) * fake_imgs, requires_grad=True
            ).to(self.device)
            prob_inter = self.discriminator(inter_x)
            gradients = torch.autograd.grad(
                outputs=prob_inter,
                inputs=inter_x,
                grad_outputs=torch.ones_like(prob_inter).to(self.device),
                create_graph=True,
                retain_graph=True,
            )[0]
            gradient_panelty = torch.mean(
                (torch.linalg.vector_norm(gradients.reshape(N, -1), dim=1) - 1) ** 2
            )

            # discriminator loss is the average of these
            d_loss = real_loss + fake_loss + self.hparams.gp_weight * gradient_panelty
            self.log("train_loss/d_loss", d_loss)
            self.log("train_log/real_logit", -real_loss)
            self.log("train_log/fake_logit", fake_loss)
            self.log("train_log/gradient_panelty", gradient_panelty)

            return d_loss

    def validation_step(self, batch, batch_idx):
        img, _ = batch
        z = torch.randn(img.shape[0], self.hparams.latent_dim).to(self.device)
        fake_imgs = self.forward(z)
        return ValidationResult(real_image=img, fake_image=fake_imgs)