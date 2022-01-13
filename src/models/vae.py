import itertools
from pathlib import Path
from turtle import forward

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf
from torch import logit, nn

from src.models.base import BaseModel
from torch import distributions

class GaussianDistribution(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, mu, target):
        dist = distributions.Normal(mu, torch.ones_like(mu))
        p_x = dist.log_prob(target).sum(dim=[1,2,3])
        return p_x
        
    def sample(self, mu):
        return mu

class BernoulliDistribution(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, target, input_normalized=True):
        if input_normalized:
            target = (target+1)/2
        prob = -F.binary_cross_entropy_with_logits(logits, target, reduction='none').sum([1, 2, 3])
        return prob

    def sample(self, logits, input_normalized=True):
        # NOTE: Actually, sampling from bernoulli will cause sharp artifacts.
        # dist = distributions.Bernoulli(logits=logits)
        # imgs = dist.sample()
        imgs = torch.sigmoid(logits)
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
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if decoder_dist == "gaussian":
            # https://stats.stackexchange.com/questions/373858/is-the-optimization-of-the-gaussian-vae-well-posed
            # Using MLL for guassian is ill-posed, so we set the variance of gaussian to be fixed
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

        post_dist = distributions.Normal(mu, torch.exp(log_sigma))
        prior_dist = distributions.Normal(torch.zeros_like(mu), torch.ones_like(log_sigma))
        samples_z = post_dist.rsample()
        # Method1: Monte-Carlo Method for KL
        kl_divergence = (post_dist.log_prob(samples_z)-prior_dist.log_prob(samples_z)).sum(dim=-1).mean(dim=0)
        # Method2: Closed-form Entropy and MC for crossentropy
        # kl_divergence = (-post_dist.entropy()-prior_dist.log_prob(samples_z)).sum(-1).mean(dim=0)
        # Method3: Closed-form KL
        # kl_divergence = -0.5 * torch.sum(1 + 2 * log_sigma - mu ** 2 - torch.exp(2 * log_sigma), dim=1).mean(dim=0)

        # decoding
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
            anchor_dist = distributions.Normal(mu[0], torch.exp(log_sigma[0]))
            anchor_samples = anchor_dist.sample(sample_shape=[64]) # 64, latent_dim
            self.log_images(self(anchor_samples), "debug/anchor_images")

            if self.hparams.latent_dim == 2:
                x = torch.tensor(np.linspace(-3, 3, 20)).type_as(imgs)
                y = torch.tensor(np.linspace(-3, 3, 20)).type_as(imgs)
                xx, yy = torch.meshgrid([x, y])
                latent = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1) # (20*20, 2)
                grid_imgs = self.decoder(latent)
                self.log_images(grid_imgs, "sample/grid_imgs", nimgs=400, nrow=20)

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
            post_dist = distributions.Normal(mu, torch.exp(log_sigma))
            self.latents.append(post_dist.rsample())
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
