import itertools
from pathlib import Path
from turtle import forward

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf
from torch import nn

from src.models.base import BaseModel
from torch import distributions

from src.utils.toy import ToyGMM

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
    def __init__(self, input_normalized=True):
        super().__init__()
        self.input_normalized = input_normalized
    
    def forward(self, logits, target):
        if self.input_normalized:
            target = (target+1)/2
        prob = -F.binary_cross_entropy_with_logits(logits, target, reduction='none').sum([1, 2, 3])
        return prob

    def sample(self, logits):
        # NOTE: Actually, sampling from bernoulli will cause sharp artifacts.
        # dist = distributions.Bernoulli(logits=logits)
        # imgs = dist.sample()
        imgs = torch.sigmoid(logits)
        if self.input_normalized:
            imgs = imgs*2-1
        return imgs

class VAE(BaseModel):
    def __init__(
        self,
        datamodule: OmegaConf = None,
        encoder: OmegaConf = None,
        decoder: OmegaConf = None,
        beta: float = 1.0,
        recon_weight: float = 1.0,
        optim: str = "adam",
        latent_dim: int = 100,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        plot_latent_traverse=False,
        decoder_dist="gaussian",
        prior_dist="normal",
        **kwargs,
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()

        if decoder_dist == "gaussian":
            # https://stats.stackexchange.com/questions/373858/is-the-optimization-of-the-gaussian-vae-well-posed
            # Using MLL for guassian is ill-posed, so we set the variance of gaussian to be fixed
            self.decoder = hydra.utils.instantiate(
                decoder, input_channel=latent_dim, output_channel=self.channels, output_act="tanh"
            )
            self.decoder_dist = GaussianDistribution()
        elif decoder_dist == "bernoulli":
            self.decoder = hydra.utils.instantiate(
                decoder, input_channel=latent_dim, output_channel=self.channels, output_act="identity"
            )
            self.decoder_dist = BernoulliDistribution(input_normalized=self.input_normalize)
        
        self.encoder = hydra.utils.instantiate(
            encoder, input_channel=self.channels, output_channel=2 * latent_dim
        )

    def get_prior_dist(self):
        if self.hparams.prior_dist == "normal":
            return distributions.MultivariateNormal(torch.zeros(self.hparams.latent_dim).to(self.device), torch.diag(torch.ones(self.hparams.latent_dim).to(self.device)))
        elif self.hparams.prior_dist == "toy_gmm":
            return ToyGMM(10, device=self.device)

    def forward(self, z=None):
        """Generate images given latent code."""
        if z == None:
            z = torch.randn(64, self.hparams.latent_dim).to(self.device)
        # decoding
        logits = self.decoder(z)
        output = self.decoder_dist.sample(logits)
        output = output.reshape(
            output.shape[0],
            self.channels,
            self.height,
            self.width,
        )
        return output

    def training_step(self, batch, batch_idx):
        imgs, labels = batch # (N, C, H, W)
        N = imgs.shape[0]

        # encoding
        z = self.encoder(imgs).reshape(N, -1)  # (N, latent_dim)
        assert z.shape == (N, 2 * self.hparams.latent_dim), f"shape of z: {z.shape}"
        mu, log_sigma = z[:, : self.hparams.latent_dim], z[:, self.hparams.latent_dim :]

        post_dist = distributions.Normal(mu, torch.exp(log_sigma))

        prior_dist = self.get_prior_dist()
        samples_z = post_dist.rsample()
        # Method1: Monte-Carlo Method for KL
        kl_divergence = (post_dist.log_prob(samples_z).sum(dim=-1)-prior_dist.log_prob(samples_z)).mean(dim=0)
        # Method2: Closed-form Entropy and MC for crossentropy
        # kl_divergence = (-post_dist.entropy()-prior_dist.log_prob(samples_z)).sum(-1).mean(dim=0)
        # Method3: Closed-form KL
        # kl_divergence = -0.5 * torch.sum(1 + 2 * log_sigma - mu ** 2 - torch.exp(2 * log_sigma), dim=-1).mean(dim=0)

        # decoding
        logits = self.decoder(samples_z)
        log_p_x_of_z = self.decoder_dist(logits, imgs).mean(dim=0)
        elbo = -self.hparams.beta*kl_divergence + self.hparams.recon_weight * log_p_x_of_z

        self.log("train_log/elbo", elbo.item())
        self.log("train_log/kl_divergence", kl_divergence.item())
        self.log("train_log/log_p_x_of_z", log_p_x_of_z.item())
        self.log("train_log/sigma", torch.exp(log_sigma).mean())

        # log sampled images
        if self.global_step % 200 == 0:
            recon_images = self.decoder_dist.sample(logits)
            sample_images = self()
            self.log_images(imgs, "train recon/source_image")
            self.log_images(recon_images, "train recon/output_image")
            self.log_images(sample_images, "train sample/output_image")
            # self.log_hist(torch.linalg.norm(torch.exp(log_sigma), dim=1), "sigma hist")

            # sample images given the first image of mini-batch 
            # NOTE: when batchnorm is applied, each batch will produce different images even the variantion is very small.
            anchor_dist = distributions.Normal(mu[0], torch.exp(log_sigma[0]))
            anchor_samples = anchor_dist.sample(sample_shape=[64]) # 64, latent_dim
            self.log_images(self(anchor_samples), "debug/anchor_images")

            # sample traverse images
            if self.hparams.plot_latent_traverse == True:
                row, col = 10, 10
                fixed_z = torch.randn(1, self.hparams.latent_dim).repeat(row*col, 1).reshape(row, col, -1).to(self.device)
                variation_z = torch.linspace(-3, 3, col)
                for i in range(row):
                    z = fixed_z # (row, col, latent)
                    z[i, :, i] = variation_z
                imgs = self.decoder(z.reshape(row*col, -1))
                self.log_images(imgs, f"sample/traverse_latents", nimgs=row*col, nrow=col)

            if self.hparams.latent_dim == 2:
                x = torch.tensor(np.linspace(-3, 3, 20)).type_as(imgs)
                y = torch.tensor(np.linspace(3, -3, 20)).type_as(imgs)
                xx, yy = torch.meshgrid([y, x])
                latent = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1) # (20*20, 2)
                grid_imgs = self.decoder(latent)
                self.log_images(grid_imgs, "sample/grid_imgs", nimgs=400, nrow=20)
                self.plot_scatter("train/latent distributions", x=samples_z[:, 0], y=samples_z[:,1], c=labels, s=2, xlim=(-3, 3), ylim=(-3, 3))
                self.plot_scatter("train/latent mu distributions", x=mu[:, 0], y=mu[:,1], c=labels, s=2, xlim=(-3, 3), ylim=(-3, 3))

                if self.hparams.prior_dist == "toy_gmm":
                    samples, labels = self.get_prior_dist().sample(2000)
                    self.plot_scatter("prior distributions", x=samples[:, 0], y=samples[:,1], c=labels, s=2, xlim=(-3, 3), ylim=(-3, 3))


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
            scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.99)
            return [opt], [scheduler]
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

        z = self.encoder(imgs)
        mu, log_sigma = z[:, : self.hparams.latent_dim], z[:, self.hparams.latent_dim :]
        post_dist = distributions.Normal(mu, torch.exp(log_sigma))
        sample_z = post_dist.rsample()
        recon_image = self(sample_z)
        sample_images = self()
        if self.hparams.latent_dim == 2:
            self.latents.append(sample_z)
            self.labels.append(label)
            self.mu.append(mu)
            self.sigma.append(torch.linalg.norm(torch.exp(log_sigma), dim=1))
        
        self.log_images(imgs, "val recon/source_image")
        self.log_images(recon_image, "val recon/output_image")
        self.log_images(sample_images, "val sample/output_image")


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
            self.plot_scatter("val/latent distributions", x=latents_array[:, 0][sort_idx], y=latents_array[:,1][sort_idx], 
                c=labels_array[sort_idx], xlim=(-3, 3), ylim=(-3, 3))
            # self.plot_scatter("val/latent mu distributions", x=mu_array[:, 0][sort_idx], y=mu_array[:,1][sort_idx],
            #     c=labels_array[sort_idx], xlim=(-3, 3), ylim=(-3, 3))
            # self.log_hist(torch.cat(self.sigma), "val/sigma hist")
