"""
Adversarial Autoencoder
https://arxiv.org/abs/1511.05644
"""
import itertools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import hydra
import torch
import torch.nn.functional as F
import torchvision
from .base import BaseModel
import torchmetrics
import io
import PIL
from torchvision.transforms import ToTensor


class AAE(BaseModel):
    def __init__(
        self,
        channels,
        width,
        height,
        encoder,
        decoder,
        netD,
        latent_dim=100,
        loss_mode="vanilla",
        lrG: float = 0.0002,
        lrD: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        input_normalize=True,
        optim="adam",
        eval_fid=False,
        recon_weight=10,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.prior = "standard_normal"
        # networks
        self.decoder = hydra.utils.instantiate(
            decoder, input_channel=latent_dim, output_channel=channels
        )
        self.encoder = hydra.utils.instantiate(
            encoder, input_channel=channels, output_channel=latent_dim
        )
        self.discriminator = hydra.utils.instantiate(
            netD, input_channel=latent_dim, output_channel=1, width=1, height=1
        )

    def forward(self, z):
        output = self.decoder(z)
        output = output.reshape(
            z.shape[0], self.hparams.channels, self.hparams.height, self.hparams.width
        )
        return output
    
    def sample_prior(self, N):
        if self.prior == "standard_normal":
            mean = np.zeros(self.hparams.latent_dim)
            var = np.diag(np.ones(self.hparams.latent_dim))
            samples = np.random.multivariate_normal(mean, var, size=(N))
        return torch.tensor(samples, dtype=torch.float32).to(self.device)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch  # (N, C, H, W)
        if self.hparams.input_normalize:
            imgs = imgs * 2 - 1

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)  # (N, latent_dim)
        z = z.type_as(imgs)

        # train encoder and decoder
        if optimizer_idx == 0:
            q_z = self.encoder(imgs) # (N, hidden_dim)
            # generate images
            generated_imgs = self.decoder(q_z)
            recon_loss = F.mse_loss(imgs, generated_imgs)
            self.log("train_loss/recon_loss", recon_loss, prog_bar=True)
            # log sampled images
            self.log_images(generated_imgs, "generated_images")

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            adv_loss = self.adversarial_loss(self.discriminator(q_z), valid)
            self.log("train_loss/adv_encoder_loss", adv_loss, prog_bar=True)
            return adv_loss+recon_loss*self.hparams.recon_weight

        # train discriminator
        if optimizer_idx == 1:
            # real loss
            N = imgs.size(0)
            valid = torch.ones(N, 1)
            valid = valid.type_as(imgs)
            real_prior = self.sample_prior(N)
            real_logit = self.discriminator(real_prior)
            real_loss = self.adversarial_loss(real_logit, valid)

            # fake loss
            fake = torch.zeros(N, 1)
            fake = fake.type_as(imgs)
            fake_logit = self.discriminator(self.encoder(imgs))
            fake_loss = self.adversarial_loss(fake_logit, fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("train_loss/d_loss", d_loss)
            self.log("train_log/real_logit", real_logit.mean())
            self.log("train_log/fake_logit", fake_logit.mean())

            return d_loss

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

    def on_validation_epoch_start(self) -> None:
        if self.hparams.eval_fid:
            self.fid = torchmetrics.FID().to(self.device)
        if self.hparams.latent_dim == 2:
            self.latents = []
            self.labels = []

    def validation_step(self, batch, batch_idx):
        imgs, label = batch
        if self.hparams.eval_fid:
            self.fid.update(self.image_float2int(imgs), real=True)
            z = self.sample_prior(imgs.shape[0]).to(self.device)
            fake_imgs = self.decoder(z)
            self.fid.update(self.image_float2int(fake_imgs), real=False)

        if self.hparams.latent_dim == 2:
            posterior_latent = self.encoder(imgs)
            self.latents.append(posterior_latent)
            self.labels.append(label)


    def on_validation_epoch_end(self):
        if self.hparams.eval_fid:
            self.log("metrics/fid", self.fid.compute())

        if self.hparams.latent_dim == 2:
            # show posterior
            latents_array = torch.cat(self.latents).cpu().numpy()
            labels_array = torch.cat(self.labels).cpu().numpy()
            self.latents = []
            self.labels = []
            plt.figure()
            plt.scatter(x=latents_array[:, 0], y=latents_array[:,1], c=labels_array, cmap="tab10")
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
            plt.title("Latent distribution")
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            visual_image = ToTensor()(PIL.Image.open(buf))
            self.logger.experiment.add_image("latent distributions", visual_image, self.global_step)

            # show prior
            latents_array = self.sample_prior(100000).cpu().numpy()
            plt.figure()
            plt.scatter(x=latents_array[:, 0], y=latents_array[:,1])
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
            plt.title("Prior Latent distribution")
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            visual_image = ToTensor()(PIL.Image.open(buf))
            self.logger.experiment.add_image("Prior latent distributions", visual_image, self.global_step)


    def adversarial_loss(self, y_hat, y):
        if self.hparams.loss_mode == "vanilla":
            return F.binary_cross_entropy_with_logits(y_hat, y)
        elif self.hparams.loss_mode == "lsgan":
            return F.mse_loss(y_hat, y)

    def configure_optimizers(self):
        lrG = self.hparams.lrG
        lrD = self.hparams.lrD
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        if self.hparams.optim == "adam":
            opt_g = torch.optim.Adam(
                itertools.chain(self.encoder.parameters(),self.decoder.parameters()), lr=lrG, betas=(b1, b2)
            )
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(), lr=lrD, betas=(b1, b2)
            )
        elif self.hparams.optim == "sgd":
            opt_g = torch.optim.SGD(itertools.chain(self.encoder.parameters(),self.decoder.parameters()) , lr=lrG)
            opt_d = torch.optim.SGD(self.discriminator.parameters(), lr=lrD)
        return [opt_g, opt_d]
