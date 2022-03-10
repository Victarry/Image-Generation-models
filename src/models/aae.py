"""
Adversarial Autoencoder
https://arxiv.org/abs/1511.05644
"""
import itertools
import numpy as np
import hydra
import torch
import torch.nn.functional as F

from src.utils.toy import ToyGMM
from src.utils.losses import adversarial_loss
from src.networks.basic import MLPEncoder
from .base import BaseModel, ValidationResult


class AAE(BaseModel):
    def __init__(
        self,
        datamodule,
        encoder,
        decoder,
        netD,
        latent_dim=100,
        loss_mode="vanilla",
        lrG: float = 0.0002,
        lrD: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        recon_weight=1,
        prior="normal",
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
        self.discriminator =  MLPEncoder(
            input_channel=latent_dim, output_channel=1, hidden_dims=[256, 256], width=1, height=1, norm_type="layer"
        )
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

        opt_g = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(),self.decoder.parameters()), lr=lrG, betas=(b1, b2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lrD, betas=(b1, b2)
        )
        return [opt_g, opt_d]

    def sample_prior(self, N):
        if self.hparams.prior == "normal":
            samples = torch.randn(N, self.hparams.latent_dim)
        elif self.hparams.prior == "toy_gmm":
            samples, _ = ToyGMM(10).sample(N) 
        return samples.to(self.device)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch  # (N, C, H, W)
        N = imgs.shape[0]
        opt_g, opt_d = self.optimizers()

        # reconstruction phase
        q_z = self.encoder(imgs) # (N, hidden_dim)
        generated_imgs = self.decoder(q_z)
        recon_loss = F.mse_loss(imgs, generated_imgs)

        self.log("train_loss/recon_loss", recon_loss)
        opt_g.zero_grad()
        self.manual_backward(recon_loss*self.hparams.recon_weight)
        opt_g.step()

        # regularization phase
        # update discriminator
        real_prior = self.sample_prior(N)
        real_logit = self.discriminator(real_prior)
        real_loss = adversarial_loss(real_logit, True, self.hparams.loss_mode)
        fake_logit = self.discriminator(self.encoder(imgs))
        fake_loss = adversarial_loss(fake_logit, False, self.hparams.loss_mode)
        d_adv_loss = (real_loss + fake_loss) / 2
        self.log("train_loss/d_loss", d_adv_loss)
        self.log("train_log/real_logit", real_logit.mean())
        self.log("train_log/fake_logit", fake_logit.mean())

        opt_d.zero_grad()
        self.manual_backward(d_adv_loss)
        opt_d.step()

        # update generator
        q_z = self.encoder(imgs)
        g_adv_loss = adversarial_loss(self.discriminator(q_z), True, self.hparams.loss_mode)
        self.log("train_loss/adv_encoder_loss", g_adv_loss)

        opt_g.zero_grad()
        self.manual_backward(g_adv_loss)
        opt_g.step()


    def validation_step(self, batch, batch_idx):
        imgs, label = batch
        z = self.encoder(imgs)
        recon_imgs = self.decoder(z)

        sample_z = self.sample_prior(imgs.shape[0])
        sample_imgs = self.decoder(sample_z)
        return ValidationResult(real_image=imgs, fake_image=sample_imgs, recon_image=recon_imgs, label=label, encode_latent=z)
