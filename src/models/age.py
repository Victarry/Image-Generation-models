"""
It Takes (Only) Two: Adversarial Generator-Encoder Networks
https://arxiv.org/abs/1704.02304
"""
from torch.nn.parameter import Parameter
from torch import distributions as D
from pathlib import Path
import numpy as np
import hydra
import torch
import torch.nn.functional as F
import torchvision
from .base import BaseModel
import torchmetrics


class AGE(BaseModel):
    def __init__(
        self,
        channels,
        width,
        height,
        encoder,
        decoder,
        lrE,
        lrG,
        latent_dim=128,
        b1: float = 0.5,
        b2: float = 0.999,
        input_normalize=True,
        optim="adam",
        eval_fid=False,
        recon_z_weight=1000,
        recon_x_weight=10,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        # networks
        self.decoder = hydra.utils.instantiate(
            decoder, input_channel=latent_dim, output_channel=channels
        )
        self.encoder = hydra.utils.instantiate(
            encoder, input_channel=channels, output_channel=latent_dim
        )

        self.prior_mu = Parameter(torch.zeros(latent_dim))
        self.prior_sigma = Parameter(torch.ones(latent_dim))

        self.prior_dist = D.Normal(self.prior_mu, self.prior_sigma)

    def forward(self, z=None):
        if z == None:
            z = torch.randn(64, self.hparams.latent_dim).to(self.device)
        output = self.decoder(z)
        output = output.reshape(
            z.shape[0], self.hparams.channels, self.hparams.height, self.hparams.width
        )
        return output
    
    def calculate_kl(self, samples: torch.Tensor):
        """Calcuate KL divergence between fitted gaussian distribution and standard normal distribution
        """
        # samples: (N, d)
        mu = samples.mean(dim=0) # (d)
        var = samples.var(dim=0) # (d)
        kl_div = (mu**2+var-torch.log(var)).mean()/2
        return kl_div

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch  # (N, C, H, W)
        N = imgs.shape[0]

        # sample noise
        # NOTE: the input to decoder and output to encoder are both in sphere latent space
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim).type_as(imgs)  # (N, latent_dim)
        norm_z = z / z.norm(dim=1, keepdim=True)

        # train encoder
        if optimizer_idx == 0:
            # divergence between prior and encoded real samples
            real_z = self.encoder(imgs).reshape(N, -1)  # (N, latent_dim)
            # NOTE: mapping the latent code to sphere
            norm_real_z = real_z/real_z.norm(dim=1, keepdim=True)
            real_kl = self.calculate_kl(norm_real_z)

            # divergence between prior and encoded generated samples
            fake_imgs = self.decoder(norm_z).detach()
            fake_z = self.encoder(fake_imgs).reshape(N, -1)
            norm_fake_z = fake_z / fake_z.norm(dim=1, keepdim=True)
            fake_kl = self.calculate_kl(norm_fake_z)

            # recon_x
            recon_imgs = self.decoder(norm_real_z)
            recon_loss = F.mse_loss(imgs, recon_imgs)


            self.log("train_loss/real_kl", real_kl)
            self.log("train_loss/fake_kl", fake_kl)
            self.log("train_loss/recon_x", recon_loss)

            if self.global_step % 200 == 0:
                # log sampled images
                self.log_images(fake_imgs, "generated_images")
                if self.hparams.latent_dim == 2:
                    self.plot_scatter("real_distribution/sample", x=real_z[:, 0], y=real_z[:, 1], c=labels)
                    self.plot_scatter("real_distribution/sample_norm", x=norm_real_z[:, 0], y=norm_real_z[:, 1], c=labels, xlim=(-1, 1), ylim=(-1, 1))

                    self.plot_scatter("fake_distribution/sample", x=fake_z[:, 0], y=fake_z[:, 1], c=labels)
                    self.plot_scatter("fake_distribution/sample_norm", x=norm_fake_z[:, 0], y=norm_fake_z[:, 1], c=labels, xlim=(-1, 1), ylim=(-1 ,1))
            # encoder try to minimize real_kl and maximize fake_kl
            return real_kl-fake_kl+self.hparams.recon_x_weight*recon_loss

        # train decoder 
        if optimizer_idx == 1:
            fake_imgs = self.decoder(norm_z)
            fake_z = self.encoder(fake_imgs).reshape(N, -1)
            norm_fake_z = fake_z / fake_z.norm(dim=1, keepdim=True)
            fake_kl = self.calculate_kl(norm_fake_z)

            # Why not also optimize recon_x, I thought it's to reduce computation.
            recon_loss = F.mse_loss(norm_fake_z, norm_z)

            self.log("train_loss/recon_z", recon_loss)
            # decoder try to minimize fake_kl
            return fake_kl + self.hparams.recon_z_weight*recon_loss

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
            fake_imgs = self.forward()
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
            sort_idx = np.argsort(labels_array)
            self.plot_scatter("Latent Distribution", x=latents_array[:, 0][sort_idx], y=latents_array[:,1][sort_idx], 
                                c=labels_array[sort_idx], xlim=(-3, 3), ylim=(-3, 3))
            self.latents = []
            self.labels = []

    def configure_optimizers(self):
        lrG = self.hparams.lrG
        lrE = self.hparams.lrE
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        if self.hparams.optim == "adam":
            opt_e = torch.optim.Adam(
                self.encoder.parameters(), lr=lrE, betas=(b1, b2)
            )
            opt_g = torch.optim.Adam(
                self.decoder.parameters(), lr=lrG, betas=(b1, b2)
            )
        elif self.hparams.optim == "sgd":
            opt_e = torch.optim.SGD(self.encoder.parameters(), lr=lrE)
            opt_g = torch.optim.SGD(self.decoder.parameters(), lr=lrG)
        return [opt_e, opt_g]
        # return [
        #     {"optimizer": opt_e, "frequency": 1},
        #     {"optimizer": opt_g, "frequency": 4},
        # ]
