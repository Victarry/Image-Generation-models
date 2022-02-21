from logging.config import valid_ident
import hydra
from numpy.core.fromnumeric import reshape
import pytorch_lightning as pl
from pytorch_lightning.core.mixins import hparams_mixin
import torchvision
import torch
import torch.nn.functional as F
from src.utils import utils
import itertools
from pathlib import Path
from omegaconf import OmegaConf
from .base import BaseModel
from torch import nn

# TODO:
# 1. sampling implementation
# 2. dive into the influence of num_embeddings and latent_dim


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, latent_dim, commitment_weight) -> None:
        super().__init__()
        self.embedding = torch.nn.parameter.Parameter(
            torch.zeros(num_embeddings, latent_dim).uniform_(
                -1 / num_embeddings, 1 / num_embeddings
            )
        )  # learned discete representation
        self.latent_dim = latent_dim
        self.commitment_weight = commitment_weight

    def forward(self, z):
        N, C, H, W = z.shape

        # (N, latent_dim, H, W) -> (N*latent_size, latent_dim)
        reshape_z = (
            z.reshape(N, self.latent_dim, -1)
            .permute(0, 2, 1)
            .reshape(-1, self.latent_dim)
        )
        # (N*latent_size, latent_dim), (K, latent_dim) -> (N*latent_size, K)
        dist = torch.cdist(reshape_z, self.embedding)

        z_index = torch.argmin(dist, dim=1)  # (N*latent_size)
        quant_z = self.embedding[z_index]  # (N*latent_size, latent_dim)
        vq_loss = F.mse_loss(reshape_z.detach(), quant_z)
        commit_loss = self.commitment_weight * F.mse_loss(reshape_z, quant_z.detach())

        quant_z = quant_z.reshape(N, H, W, C).permute(0, 3, 1, 2)

        return quant_z, vq_loss, commit_loss


class VQVAE(BaseModel):
    def __init__(
        self,
        datamodule,
        encoder: OmegaConf = None,
        decoder: OmegaConf = None,
        latent_dim=100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        num_embeddings: int = 512,
        beta: float = 0.25,
        optim="adam",
        **kwargs,
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()

        self.decoder = hydra.utils.instantiate(
            decoder, input_channel=latent_dim, output_channel=self.channels
        )
        self.encoder = hydra.utils.instantiate(
            encoder, input_channel=self.channels, output_channel=latent_dim
        )
        self.vector_quntizer = VectorQuantizer(num_embeddings, latent_dim, beta)

        self.latent_w = self.hparams.width // 4
        self.latent_h = self.hparams.height // 4
        self.latent_size = self.latent_h * self.latent_w

    def forward(self, imgs):
        """
        Directly sample from embeddings will not produce meaningful images.
        """
        if self.hparams.input_normalize:
            imgs = imgs * 2 - 1
        z = self.encoder(imgs)
        quant_z, _, _ = self.vector_quntizer(z)
        output = self.decoder(quant_z)
        output = output.reshape(
            output.shape[0],
            self.channels,
            self.height,
            self.width,
        )
        return output

    def on_train_epoch_end(self):
        result_path = Path("results")
        result_path.mkdir(parents=True, exist_ok=True)
        if hasattr(self, "val_imgs"):
            val_imgs = self.val_imgs
        else:
            val_imgs = self.val_imgs = next(
                iter(self.trainer.datamodule.test_dataloader())
            )[0].to(self.device)
        imgs = self.forward(val_imgs)
        grid = self.get_grid_images(imgs)
        torchvision.utils.save_image(grid, result_path / f"{self.current_epoch}.jpg")

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        if self.hparams.input_normalize:
            imgs = imgs * 2 - 1

        ## Encoding
        encoder_z = self.encoder(imgs)  # (N, latent_dim, latent_w, latent_h)

        ##  Vector Quantization
        quant_z, vq_loss, commit_loss = self.vector_quntizer(encoder_z)

        ## Decoding
        # this will feed value of z to decoder, and backward gradient to encoder_z instead of z,
        # such the encoder_z and encoder can be optimized
        decoder_z = encoder_z + (quant_z - encoder_z).detach()
        fake_imgs = self.decoder(decoder_z)
        fake_imgs = fake_imgs.reshape(
            -1, self.hparams.channels, self.hparams.height, self.hparams.width
        )
        recon_loss = F.mse_loss(fake_imgs, imgs)

        total_loss = recon_loss + vq_loss + self.hparams.beta * commit_loss

        self.log("train_loss/vq_loss", vq_loss)
        self.log("train_loss/recon_loss", recon_loss)
        self.log("train_loss/commit_loss", commit_loss)

        # log sampled images
        if self.global_step % 50 == 0:
            self.log_images(imgs, "recon/source_image")
            self.log_images(fake_imgs, "recon/output_image")

        return total_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt = torch.optim.Adam(
            itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters(),
                self.vector_quntizer.parameters(),
            ),
            lr=lr,
            betas=(b1, b2),
        )
        return opt
