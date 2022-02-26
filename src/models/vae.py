from hydra.utils import instantiate
import torch
from omegaconf import OmegaConf

from src.models.base import BaseModel, ValidationResult
from torch import distributions
from src.utils.distributions import GaussianDistribution
from src.utils.losses import normal_kld


class VAE(BaseModel):
    def __init__(
        self,
        datamodule: OmegaConf = None,
        encoder: OmegaConf = None,
        decoder: OmegaConf = None,
        latent_dim: int = 100,
        beta: float = 1.0,
        recon_weight: float = 1.0,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.999,
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()

        self.decoder = instantiate(decoder, input_channel=latent_dim, output_channel=self.channels)
        self.encoder = instantiate(encoder, input_channel=self.channels, output_channel=2 * latent_dim)

    def forward(self, z):
        """Generate images given latent code."""
        output = self.decoder(z)
        output = output.reshape(output.shape[0], self.channels, self.height, self.width)
        return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=(b1, b2))
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.99)
        return [opt], [scheduler]
    
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
        imgs, labels = batch # (N, C, H, W)
        N = imgs.shape[0]

        mu, log_sigma, z, recon_imgs = self.vae(imgs)
        kld = normal_kld(mu, log_sigma)
        log_p_x_of_z = GaussianDistribution(recon_imgs).prob(imgs).mean(dim=0)
        elbo = -self.hparams.beta*kld + self.hparams.recon_weight * log_p_x_of_z

        self.log("train_log/elbo", elbo)
        self.log("train_log/kl_divergence", kld)
        self.log("train_log/log_p_x_of_z", log_p_x_of_z)
        return -elbo 
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        N = imgs.shape[0]
        mu, log_sigma, z, recon_imgs = self.vae(imgs)
        log_p_x_of_z = GaussianDistribution(recon_imgs).prob(imgs).mean(dim=0)

        fake_imgs = self.sample(N)
        self.log("val_log/log_p_x_of_z", log_p_x_of_z)
        return ValidationResult(real_image=imgs, fake_image=fake_imgs, 
                    recon_image=recon_imgs, label=labels, encode_latent=z)