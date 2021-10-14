import hydra
import pytorch_lightning as pl
import torchvision
import torch
import torch.nn.functional as F
from src.utils import utils
import itertools
from pathlib import Path
from omegaconf import OmegaConf


class VAE(pl.LightningModule):
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
        imgs, _ = batch
        N = imgs.shape[0]
        if self.hparams.input_normalize:
            imgs = imgs * 2 - 1

        # encoding
        z = self.encoder(imgs).reshape(N, -1)  # (N, latent_dim)
        assert z.shape == (N, 2 * self.hparams.latent_dim), f"shape of z: {z.shape}"
        mu, log_sigma = z[:, : self.hparams.latent_dim], z[:, self.hparams.latent_dim :]

        # note the negative mark
        reg_loss = (
            -0.5 * torch.sum(1 + 2 * log_sigma - mu ** 2 - torch.exp(2 * log_sigma)) / N
        )

        # reparameterization
        noise = torch.randn(N, self.hparams.latent_dim).type_as(imgs)
        # numerical problem
        samples = noise * torch.exp(log_sigma) + mu

        # decoding
        fake_imgs = self.decoder(samples)
        fake_imgs = fake_imgs.reshape(
            -1, self.hparams.channels, self.hparams.height, self.hparams.width
        )
        # NOTE: use sum instead of mean to support adequate gradient for reconstruction, otherwise output image will converge to a single mode
        recon_loss = F.mse_loss(fake_imgs, imgs, reduction="sum") / N

        total_loss = self.hparams.reg_weight * reg_loss + recon_loss

        self.log("train_loss/reg_loss", reg_loss.item())
        self.log("train_loss/recon_loss", recon_loss.item())

        # log sampled images
        if self.global_step % 50 == 0:
            sample_images = self()
            self.log_images(imgs, "recon/source_image")
            self.log_images(fake_imgs, "recon/output_image")
            self.log_images(sample_images, "sample/output_image")

        return total_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=lr,
            betas=(b1, b2),
        )
        return opt
