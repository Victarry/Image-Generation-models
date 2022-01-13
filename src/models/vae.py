import itertools
from pathlib import Path
from turtle import forward

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf
from torch import bernoulli, logit, nn

from src.models.base import BaseModel

def reparameterize(mu, log_sigma, n_samples=1):
    # reparameterization
    if n_samples > 1:
        noise = torch.randn(*[n_samples, *mu.shape]).type_as(mu)
        samples = noise * torch.exp(log_sigma) + mu
        samples = samples.reshape(-1, *mu.shape[1:])
    else:
        noise = torch.randn_like(mu).type_as(mu)
        samples = noise * torch.exp(log_sigma) + mu
    return samples

def standard_normal_log_prob(z):
    return -0.5 * np.log(2 * np.pi) - torch.pow(z, 2) / 2

def normal_log_prob(mu, sigma, z):
    # NOTE: this may be positve, since it's continous distribution
    var = torch.pow(sigma, 2)
    log_prob = -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - mu, 2) / (2 * var)
    return log_prob

def kl_standard_normal(mu, log_sigma, closed_form=True):
    # Closed form KL divergence between a isotropic normal and standard normal distribution
    if closed_form:
        return -0.5 * torch.sum(1 + 2 * log_sigma - mu ** 2 - torch.exp(2 * log_sigma), dim=1)
    # Monte Carlo Estimation of KL
    else:
        z = reparameterize(mu, log_sigma)
        kl_div = -normal_log_prob(mu, torch.exp(log_sigma), z) - standard_normal_log_prob(z)
        return kl_div.sum(dim=-1)

def bernoulli_log_prob(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)

class NormalLogProb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loc, scale, z):
        var = torch.pow(scale, 2)
        return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (2 * var)

# class GaussianDistribution(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, logits, target):
#         mu, log_sigma = torch.chunk(logits, chunks=2, dim=1)
#         prob_target = normal_log_prob(mu, torch.exp(log_sigma), target)
#         return prob_target.sum(dim=[1, 2, 3])
        
#     def sample(self, logits, return_mean=False):
#         mu, log_sigma = torch.chunk(logits, chunks=2, dim=1)
#         if return_mean:
#             return mu
#         else:
#             return reparameterize(mu, log_sigma)

class GaussianDistribution(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, target):
        return -F.mse_loss(logits, target, reduction="none").sum([1, 2, 3])
        
    def sample(self, logits):
        return logits

class BernoulliDistribution(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, target, input_normalized=True):
        if input_normalized:
            target = (target + 1) / 2
        return -F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum([1, 2, 3])

    def sample(self, logits, input_normalized=True):
        imgs = torch.bernoulli(torch.sigmoid(logits))
        if input_normalized:
            imgs = imgs*2-1
        return imgs

class VAE(BaseModel):
    def __init__(
        self,
        channels: int = 3,
        width: int = 64,
        height: int = 64,
        encoder: OmegaConf = None,
        decoder: OmegaConf = None,
        beta: float = 1.0,
        recon_weight: float = 1.0,
        optim: str = "adam",
        latent_dim: int =100,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        input_normalize=True,
        decoder_dist="gaussian",
        closed_kl=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if decoder_dist == "gaussian":
            # https://stats.stackexchange.com/questions/373858/is-the-optimization-of-the-gaussian-vae-well-posed
            # Using MLL for guassian is ill-posed, so we just use MSE as an approximate
            self.decoder = hydra.utils.instantiate(
                decoder, input_channel=latent_dim, output_channel=channels, output_act="tanh"
            )
            self.decoder_dist = GaussianDistribution()
        elif decoder_dist == "bernoulli":
            self.decoder = hydra.utils.instantiate(
                decoder, input_channel=latent_dim, output_channel=channels, output_act="identity"
            )
            self.decoder_dist = BernoulliDistribution()
        self.encoder = hydra.utils.instantiate(
            encoder, input_channel=channels, output_channel=2 * latent_dim
        )

    def forward(self, z=None):
        """Generate images given latent code."""
        if z == None:
            z = torch.randn(64, self.hparams.latent_dim).to(self.device)
        # decoding
        logits = self.decoder(z)
        output = self.decoder_dist.sample(logits)
        output = output.reshape(
            output.shape[0],
            self.hparams.channels,
            self.hparams.height,
            self.hparams.width,
        )
        return output

    def training_step(self, batch, batch_idx):
        imgs, _ = batch # (N, C, H, W)
        N = imgs.shape[0]
        if self.hparams.input_normalize:
            imgs = imgs * 2 - 1

        # encoding
        z = self.encoder(imgs).reshape(N, -1)  # (N, latent_dim)
        assert z.shape == (N, 2 * self.hparams.latent_dim), f"shape of z: {z.shape}"
        mu, log_sigma = z[:, : self.hparams.latent_dim], z[:, self.hparams.latent_dim :]

        # note the negative mark
        kl_divergence = kl_standard_normal(mu, log_sigma, closed_form=self.hparams.closed_kl).mean(dim=0)

        # decoding
        samples_z = reparameterize(mu, log_sigma)
        logits = self.decoder(samples_z)
        log_p_x_of_z = self.decoder_dist(logits, imgs).mean(dim=0)
        elbo = -self.hparams.beta*kl_divergence + self.hparams.recon_weight * log_p_x_of_z

        self.log("train_log/elbo", elbo.item())
        self.log("train_log/kl_divergence", kl_divergence.item())
        self.log("train_log/log_p_x_of_z", log_p_x_of_z.item())
        self.log("train_log/sigma", torch.exp(log_sigma).mean())

        # log sampled images
        if self.global_step % 50 == 0:
            recon_images = self.decoder_dist.sample(logits)
            sample_images = self()
            self.log_images(imgs, "recon/source_image")
            self.log_images(recon_images, "recon/output_image")
            self.log_images(sample_images, "sample/output_image")

            # sample images given the first image of mini-batch 
            # NOTE: when batchnorm is applied, each batch will produce different images even the variantion is very small.
            anchor_mu = mu[0:1]
            anchor_log_sigma = log_sigma[0:1]
            anchor_samples = reparameterize(anchor_mu, anchor_log_sigma, n_samples=64)
            anchor_samples[0] = anchor_mu
            self.logger.experiment.add_text("anchor_latents", str(anchor_samples), self.global_step)
            self.logger.experiment.add_text("sample_sigma", str(torch.exp(anchor_log_sigma)), self.global_step)
            self.log_images(self(anchor_samples), "debug/anchor_images")

        return -elbo 
    
    def on_train_epoch_end(self):
        result_path = Path("results")
        result_path.mkdir(parents=True, exist_ok=True)
        if hasattr(self, "z"):
            z = self.z
        else:
            self.z = z = torch.randn(64, self.hparams.latent_dim).to(self.device)
        imgs = self(z)
        grid = self.get_grid_images(imgs)
        torchvision.utils.save_image(grid, result_path / f"{self.current_epoch}.jpg")

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        if self.hparams.optim == 'adam':
            opt = torch.optim.Adam(
                itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
                lr=lr,
                betas=(b1, b2),
            )
        elif self.hparams.optim == 'sgd':
            opt = torch.optim.SGD(
                itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
                lr=lr
            )
        return opt

    def on_validation_epoch_start(self) -> None:
        if self.hparams.latent_dim == 2:
            self.latents = []
            self.labels = []
            self.mu = []
            self.sigma = []

    def validation_step(self, batch, batch_idx):
        imgs, label = batch
        if self.hparams.latent_dim == 2:
            z = self.encoder(imgs)
            mu, log_sigma = z[:, : self.hparams.latent_dim], z[:, self.hparams.latent_dim :]
            self.latents.append(reparameterize(mu, log_sigma))
            self.labels.append(label)
            self.mu.append(mu)
            self.sigma.append(torch.linalg.norm(torch.exp(log_sigma), dim=1))
    
    def on_validation_epoch_end(self):
        if self.hparams.latent_dim == 2:
            # show posterior
            latents_array = torch.cat(self.latents).cpu().numpy()
            labels_array = torch.cat(self.labels).cpu().numpy()
            mu_array = torch.cat(self.mu).cpu().numpy()
            sigma_array = torch.cat(self.sigma).cpu().numpy()
            sort_idx = np.argsort(labels_array)
            self.latents = []
            self.labels = []
            self.plot_scatter("latent distributions", x=latents_array[:, 0][sort_idx], y=latents_array[:,1][sort_idx],s=sigma_array[sort_idx], 
                c=labels_array[sort_idx], xlim=(-3, 3), ylim=(-3, 3))
            self.plot_scatter("latent mu distributions", x=mu_array[:, 0][sort_idx], y=mu_array[:,1][sort_idx], s=1, 
                c=labels_array[sort_idx], xlim=(-3, 3), ylim=(-3, 3))
