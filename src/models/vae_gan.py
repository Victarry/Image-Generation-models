"""Autoencoding beyond pixels using a learned similarity metric"""
import hydra
import pytorch_lightning as pl
import torchvision
import torch
import torch.nn.functional as F
from src.utils import utils
import itertools
from pathlib import Path
from omegaconf import OmegaConf

"""
1. Remove g_adv_loss and replace recon_feature_loss with pixel_recon_loss, this degrades to vanilla vae.
2. Remove reg_loss and recon loss, this degrades to gan.
"""


class VAEGAN(pl.LightningModule):
    def __init__(
        self,
        channels: int = 3,
        width: int = 64,
        height: int = 64,
        encoder: OmegaConf = None,
        decoder: OmegaConf = None,
        reg_weight: float = 1.0,
        latent_dim=100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        recon_wegiht=1,
        input_normalize=True,
        optim="adam",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.decoder = hydra.utils.instantiate(
            decoder, input_channel=latent_dim, output_channel=channels
        )
        self.encoder = hydra.utils.instantiate(
            encoder, input_channel=channels, output_channel=2 * latent_dim
        )
        self.discriminator = hydra.utils.instantiate(
            encoder, input_channel=channels, output_channel=1, return_features=True
        )
        self.automatic_optimization = False  # disable automatic optimization

    def forward(self):
        noise = torch.randn(64, self.hparams.latent_dim).to(self.device)

        # decoding
        output = self.decoder(noise)
        output = output.reshape(
            output.shape[0],
            self.hparams.channels,
            self.hparams.height,
            self.hparams.width,
        )
        return output

    def adversarial_loss(self, y_hat, y):
        if self.hparams.loss_mode == "vanilla":
            return F.binary_cross_entropy_with_logits(y_hat, y)
        elif self.hparams.loss_mode == "lsgan":
            return F.mse_loss(y_hat, y)

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

    def get_grid_images(self, imgs):
        imgs = imgs.reshape(
            -1, self.hparams.channels, self.hparams.height, self.hparams.width
        )
        if self.hparams.input_normalize:
            grid = torchvision.utils.make_grid(
                imgs[:64], normalize=True, value_range=(-1, 1)
            )
        else:
            grid = torchvision.utils.make_grid(imgs[:64], normalize=False)
        return grid

    def log_images(self, imgs, name):
        grid = self.get_grid_images(imgs)
        self.logger.experiment.add_image(name, grid, self.global_step)

    def training_step(self, batch, batch_idx):
        encoder_optim, decoder_optim, discriminator_optim = self.optimizers()
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        discriminator_optim.zero_grad()

        imgs, _ = batch
        N = imgs.shape[0]
        if self.hparams.input_normalize:
            imgs = imgs * 2 - 1
        # encoding
        z = self.encoder(imgs).reshape(N, -1)  # (N, latent_dim)
        assert z.shape == (N, 2 * self.hparams.latent_dim), f"shape of z: {z.shape}"
        mu, log_sigma = z[:, : self.hparams.latent_dim], z[:, self.hparams.latent_dim :]

        # reparameterization
        noise = torch.randn(N, self.hparams.latent_dim).to(self.device)
        samples = noise * torch.exp(log_sigma) + mu
        random_samples = torch.randn(N, self.hparams.latent_dim).to(self.device)

        # decoding
        recon_imgs = self.decoder(samples)
        fake_imgs = self.decoder(random_samples)

        # discrimination
        real_logit, real_features = self.discriminator(imgs)
        recon_logit, recon_features = self.discriminator(recon_imgs)
        fake_logit, _ = self.discriminator(fake_imgs)

        real_label = torch.ones_like(real_logit).to(self.device)
        fake_label = torch.zeros_like(fake_logit).to(self.device)

        # optimization of discriminator
        d_adv_loss = (
            self.adversarial_loss(real_logit, real_label)
            + self.adversarial_loss(fake_logit, fake_label)
            # NOTE: remove recon logit from d_adv_loss may improve the sample quality
            + self.adversarial_loss(recon_logit, fake_label)
        )
        d_adv_loss.backward(
            inputs=list(self.discriminator.parameters()), retain_graph=True
        )

        # encoder optimization
        reg_loss = (
            -0.5 * torch.sum(1 + 2 * log_sigma - mu ** 2 - torch.exp(2 * log_sigma)) / N
        )
        reg_loss.backward(retain_graph=True)
        feature_recon_loss = (
            F.mse_loss(real_features, recon_features, reduction="sum") / N
        )
        feature_recon_loss.backward(
            inputs=list([*self.encoder.parameters(), *self.decoder.parameters()]),
            retain_graph=True,
        )
        for p in self.decoder.parameters():
            p.grad *= self.hparams.recon_weight

        # decoder
        g_adv_loss = self.adversarial_loss(
            fake_logit, real_label
        ) + self.adversarial_loss(recon_logit, real_label)
        g_adv_loss.backward(inputs=list(self.decoder.parameters()))

        decoder_optim.step()
        encoder_optim.step()
        discriminator_optim.step()

        self.log("train_loss/reg_loss", reg_loss)
        self.log("train_loss/feature_recon_loss", feature_recon_loss)
        self.log("train_loss/d_adv_loss", d_adv_loss)
        self.log("train_loss/g_adv_loss", g_adv_loss)
        self.log("train_log/real_logit", torch.mean(real_logit))
        self.log("train_log/fake_logit", torch.mean(fake_logit))
        self.log("train_log/recon_logit", torch.mean(recon_logit))

        # log sampled images
        if self.global_step % 50 == 0:
            sample_images = self()
            self.log_images(imgs, "recon/source_image")
            self.log_images(recon_imgs, "recon/output_image")
            self.log_images(fake_imgs, "sample/output_image")

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        encoder_optim = torch.optim.Adam(
            self.encoder.parameters(),
            lr=lr,
            betas=(b1, b2),
        )
        decoder_optim = torch.optim.Adam(
            self.decoder.parameters(),
            lr=lr,
            betas=(b1, b2),
        )
        discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(b1, b2),
        )
        return encoder_optim, decoder_optim, discriminator_optim
