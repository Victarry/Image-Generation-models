"""Autoencoding beyond pixels using a learned similarity metric"""
import itertools
import hydra
import torch
from omegaconf import OmegaConf
import torch.distributions as D

from .base import BaseModel, ValidationResult
from src.utils.distributions import get_decode_dist
from src.networks.basic import MLPEncoder
from src.utils.losses import adversarial_loss, normal_kld

def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)
    return torch.cat(perm_z, 1)

class FactorVAE(BaseModel):
    def __init__(
        self,
        datamodule,
        encoder: OmegaConf = None,
        decoder: OmegaConf = None,
        loss_mode: str = 'lsgan',
        adv_weight: float = 1,
        latent_dim=10,
        lr: float = 0.0002,
        lrD: float = 0.0001,
        ae_b1: float = 0.9, # adam parameter for encoder and decoder
        ae_b2: float = 0.999,
        adv_b1: float = 0.5, # adam paramter for discriminator
        adv_b2: float = 0.999,
        decoder_dist="gaussian"
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()

        self.decoder = hydra.utils.instantiate(decoder, input_channel=latent_dim, output_channel=self.channels, output_act=self.output_act)
        self.decoder_dist = get_decode_dist(decoder_dist)

        self.encoder = hydra.utils.instantiate(encoder, input_channel=self.channels, output_channel=latent_dim*2)
        self.netD = MLPEncoder(input_channel=latent_dim, hidden_dims=[256, 256], output_channel=1, width=1, height=1)
        self.automatic_optimization = False  # disable automatic optimization

    def forward(self, z):
        output = self.decoder(z)
        output = output.reshape(output.shape[0], self.channels, self.height, self.width)
        return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        lrD = self.hparams.lrD
        ae_b1 = self.hparams.ae_b1
        ae_b2 = self.hparams.ae_b2
        adv_b1 = self.hparams.adv_b1
        adv_b2 = self.hparams.adv_b2

        ae_optim = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=lr, betas=(ae_b1, ae_b2),)
        discriminator_optim = torch.optim.Adam(self.netD.parameters(), lr=lrD, betas=(adv_b1, adv_b2),)
        return ae_optim, discriminator_optim
    
    def reparameterize(self, mu, log_sigma):
        post_dist = D.Normal(mu, torch.exp(log_sigma))
        samples_z = post_dist.rsample()
        return samples_z
    
    def encode(self, imgs):
        z_ = self.encoder(imgs)  # (N, latent_dim)
        mu, log_sigma = torch.chunk(z_, chunks=2, dim=1)
        z = self.reparameterize(mu, log_sigma)
        return z, mu, log_sigma

    def vae(self, imgs):
        z, mu, log_sigma = self.encode(imgs)
        recon_imgs = self.decoder(z)
        return z, recon_imgs, mu, log_sigma, 

    def training_step(self, batch, batch_idx):
        ae_optim, discriminator_optim = self.optimizers()
        imgs, _ = batch
        imgs1, imgs2 = torch.chunk(imgs, 2, dim=0)

        # auto-encoding
        z1_samples, recon_imgs, mu, log_sigma = self.vae(imgs1)

        reg_loss = normal_kld(mu, log_sigma)
        recon_loss = -self.decoder_dist.prob(recon_imgs, imgs1).mean(dim=0)

        fake_logit = self.netD(z1_samples)
        g_adv_loss = adversarial_loss(fake_logit, target_is_real=True, loss_mode=self.hparams.loss_mode)
        encoder_loss = recon_loss + reg_loss + self.hparams.adv_weight * g_adv_loss

        ae_optim.zero_grad()
        encoder_loss.backward(retain_graph=True)
        ae_optim.step()

        # # discrimination
        z2_samples, _, _ = self.encode(imgs2) # (N, latent_dim)
        perm_z = permute_dims(z2_samples)

        real_logit = self.netD(perm_z)
        d_adv_loss = adversarial_loss(real_logit, True, self.hparams.loss_mode) + adversarial_loss(fake_logit, False, self.hparams.loss_mode)

        discriminator_optim.zero_grad()
        d_adv_loss.backward(inputs=list(self.netD.parameters()))
        discriminator_optim.step()

        self.log("train_loss/reg_loss", reg_loss)
        self.log("train_loss/recon_loss", recon_loss, prog_bar=True)
        self.log("train_loss/d_adv_loss", d_adv_loss)
        self.log("train_loss/g_adv_loss", g_adv_loss)
        self.log("train_log/real_logit", torch.mean(real_logit))
        self.log("train_log/fake_logit", torch.mean(fake_logit))
    
    def validation_step(self, batch, batch_idx):
        imgs, label = batch
        N = imgs.shape[0]

        z, recon_imgs, mu, log_sigma = self.vae(imgs)
        fake_image = self.sample(N)
        return ValidationResult(real_image=imgs, fake_image=fake_image, 
                recon_image=recon_imgs, encode_latent=z, label=label)