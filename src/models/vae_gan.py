"""Autoencoding beyond pixels using a learned similarity metric"""
import itertools
import hydra
import pytorch_lightning as pl
import torchvision
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf
from .base import BaseModel, ValidationResult
from src.utils.losses import adversarial_loss, normal_kld
from torch import distributions

class VAEGAN(BaseModel):
    def __init__(
        self,
        datamodule,
        encoder: OmegaConf = None,
        decoder: OmegaConf = None,
        latent_dim=100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        # reconstruction weight in discriminator feature space, first tune this parameter if performace is unsatifactory.
        recon_weight: float = 1e-4, 
        loss_mode: str = "vanilla"
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()

        self.decoder = hydra.utils.instantiate(decoder, input_channel=latent_dim, output_channel=self.channels)
        self.encoder = hydra.utils.instantiate(encoder, input_channel=self.channels, output_channel=2 * latent_dim)
        self.netD = hydra.utils.instantiate(encoder, input_channel=self.channels, output_channel=1, return_features=True)
        self.automatic_optimization = False

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        optim_ae = torch.optim.Adam(itertools.chain(self.encoder.parameters(), 
                        self.decoder.parameters()), lr=lr, betas=(b1, b2))
        optim_d = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(b1, b2))

        return optim_ae, optim_d

    def forward(self, z):
        output = self.decoder(z)
        output = output.reshape(output.shape[0], self.channels, self.height, self.width)
        return output
    
    def reparameterize(self, mu, log_sigma):
        post_dist = distributions.Normal(mu, torch.exp(log_sigma))
        samples_z = post_dist.rsample()
        return samples_z

    def vae(self, imgs):
        z_ = self.encoder(imgs)  # (N, latent_dim)
        mu, log_sigma = torch.chunk(z_, chunks=2, dim=1)
        z = self.reparameterize(mu, log_sigma)
        recon_imgs = self.decoder(z)
        return mu, log_sigma, z, recon_imgs

    def training_step(self, batch, batch_idx):
        optim_ae, optim_d = self.optimizers()

        imgs, _ = batch
        N = imgs.shape[0]

        mu, log_sigma, infered_z, recon_imgs = self.vae(imgs)
        prior_z = torch.randn(N, self.hparams.latent_dim).to(self.device)
        fake_imgs = self.decoder(prior_z)

        reg_loss = normal_kld(mu, log_sigma)

        fake_logit, fake_features = self.netD(fake_imgs)
        real_logit, real_features = self.netD(imgs)
        recon_logit, recon_features = self.netD(recon_imgs)
        feature_recon_loss = F.mse_loss(real_features, recon_features, reduction="sum") / N
        # NOTE: this paper says also use recon samples as , 
        # but the official code doesn't use recon images as negative samples
        g_adv_loss = adversarial_loss(fake_logit, True)

        optim_ae.zero_grad()
        self.manual_backward(reg_loss+feature_recon_loss, retain_graph=True)
        for p in self.decoder.parameters():
            p.grad *= self.hparams.recon_weight
        # encoder is not optimized w.r.t. GAN loss
        self.manual_backward(g_adv_loss, inputs=list(self.decoder.parameters()), retain_graph=True)
        optim_ae.step()

        d_adv_loss = adversarial_loss(real_logit, True) + adversarial_loss(fake_logit, False)
        optim_d.zero_grad()
        self.manual_backward(d_adv_loss, inputs=list(self.netD.parameters()))
        optim_d.step()

        self.log("train_loss/reg_loss", reg_loss)
        self.log("train_loss/feature_recon_loss", feature_recon_loss)
        self.log("train_loss/g_adv_loss", g_adv_loss)
        self.log("train_loss/d_adv_loss", d_adv_loss)
        self.log("train_log/real_logit", torch.mean(real_logit))
        self.log("train_log/fake_logit", torch.mean(fake_logit))
        self.log("train_log/recon_logit", torch.mean(recon_logit))

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        N = imgs.shape[0]
        mu, log_sigma, z, recon_imgs = self.vae(imgs)
        fake_imgs = self.sample(N)
        val_mse = F.mse_loss(imgs, recon_imgs)
        self.log("val_log/van_mse", val_mse)

        return ValidationResult(real_image=imgs, fake_image=fake_imgs, 
                    recon_image=recon_imgs, label=labels, encode_latent=z)