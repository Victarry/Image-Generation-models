"""
It Takes (Only) Two: Adversarial Generator-Encoder Networks
https://arxiv.org/abs/1704.02304
"""
import hydra
import torch
import torch.nn.functional as F
from .base import BaseModel, ValidationResult


class AGE(BaseModel):
    def __init__(
        self,
        datamodule,
        encoder,
        decoder,
        lrE,
        lrG,
        latent_dim=128,
        b1: float = 0.5,
        b2: float = 0.999,
        e_recon_z_weight=1000,
        e_recon_x_weight=0,
        g_recon_z_weight=0,
        g_recon_x_weight=10,
        norm_z=True,
        drop_lr_epoch=20,
        g_updates=2, # number of decoder iterates relative to encoder
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()
        # networks
        self.decoder = hydra.utils.instantiate(
            decoder, input_channel=latent_dim, output_channel=self.channels
        )
        self.encoder = hydra.utils.instantiate(
            encoder, input_channel=self.channels, output_channel=latent_dim
        )

    def forward(self, z):
        output = self.decoder(z)
        output = output.reshape(
            z.shape[0], self.channels, self.height, self.width
        )
        return output

    def configure_optimizers(self):
        lrG = self.hparams.lrG
        lrE = self.hparams.lrE
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        lambda_func = lambda epoch: 0.5 ** (epoch // self.hparams.drop_lr_epoch)
        opt_e = torch.optim.Adam(self.encoder.parameters(), lr=lrE, betas=(b1, b2))
        e_scheduler = torch.optim.lr_scheduler.LambdaLR(opt_e, lr_lambda=lambda_func)

        opt_g = torch.optim.Adam(self.decoder.parameters(), lr=lrG, betas=(b1, b2))
        g_scheduler = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lambda_func)
        return [
            {"optimizer": opt_e, "frequency": 1, "scheduler": e_scheduler},
            {"optimizer": opt_g, "frequency": self.hparams.g_updates, "scheduler": g_scheduler}
        ]

    def calculate_kl(self, samples: torch.Tensor, return_state=False):
        """Calcuate KL divergence between fitted gaussian distribution and standard normal distribution
        """
        assert samples.dim() == 2
        mu = samples.mean(dim=0) # (d)
        var = samples.var(dim=0) # (d)
        kl_div = (mu**2+var-torch.log(var)).mean()/2
        if return_state:
            return kl_div, mu.mean(), var.mean()
        else:
            return kl_div

    def encode(self, imgs):
        N = imgs.shape[0]
        z = self.encoder(imgs).reshape(N, -1)
        if self.hparams.norm_z:
            z = F.normalize(z)
        return z

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch  # (N, C, H, W)
        N = imgs.shape[0]

        # sample noise
        # NOTE: the input to decoder and output to encoder are both in sphere latent space
        # This is useful to prevent kl divergence from explosion
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim).type_as(imgs)  # (N, latent_dim)
        if self.hparams.norm_z:
            z = F.normalize(z)

        # train encoder
        if optimizer_idx == 0:
            # divergence between prior and encoded real samples
            real_z = self.encode(imgs)  # (N, latent_dim)
            real_kl, real_mu, real_var = self.calculate_kl(real_z, return_state=True)

            # divergence between prior and encoded generated samples
            fake_imgs = self.decoder(z)
            fake_z = self.encode(fake_imgs)
            fake_kl, fake_mu, fake_var = self.calculate_kl(fake_z, return_state=True)

            recon_x_loss = 0
            if self.hparams.e_recon_x_weight > 0:
                # recon_x, also prevent encoder mode collapse
                recon_imgs = self.decoder(real_z)
                recon_x_loss = F.mse_loss(imgs, recon_imgs, reduction="mean")
                self.log("train_loss/recon_x", recon_x_loss)

            recon_z_loss = 0
            if self.hparams.e_recon_z_weight > 0:
                recon_z_loss = 1-F.cosine_similarity(fake_z, z).mean()

            total_e_loss = real_kl-fake_kl+self.hparams.e_recon_x_weight*recon_x_loss+self.hparams.e_recon_z_weight*recon_z_loss

            self.log("train_loss/real_kl", real_kl)
            self.log("train_loss/fake_kl", fake_kl)
            self.log("train_loss/total_e_loss", total_e_loss)
            self.log("train_log/real_mu", real_mu)
            self.log("train_log/real_var", real_var)
            self.log("train_log/fake_mu", fake_mu)
            self.log("train_log/fake_var", fake_var)
            return total_e_loss

        # train decoder 
        if optimizer_idx == 1:
            fake_imgs = self.decoder(z)
            fake_z = self.encode(fake_imgs)
            fake_kl = self.calculate_kl(fake_z)

            # recon_loss = 1-F.cosine_similarity(fake_z, z).mean()
            recon_z_loss = 0
            if self.hparams.g_recon_z_weight > 0:
                recon_z_loss = F.mse_loss(fake_z, z)
            
            recon_x_loss = 0
            if self.hparams.g_recon_x_weight > 0:
                real_z = self.encode(imgs)
                recon_x = self.decoder(real_z)
                recon_x_loss = F.mse_loss(imgs, recon_x)

            total_g_loss = fake_kl + self.hparams.g_recon_z_weight*recon_z_loss + self.hparams.g_recon_x_weight*recon_x_loss

            self.log("train_loss/g_recon_z", recon_z_loss)
            self.log("train_loss/g_loss", total_g_loss)
            return total_g_loss

    def validation_step(self, batch, batch_idx):
        img, _ = batch
        z = torch.randn(img.shape[0], self.hparams.latent_dim).to(self.device)
        if self.hparams.norm_z:
            z = F.normalize(z) 

        fake_img = self.forward(z)
        encode_z = self.encode(img)
        recon_img = self.decoder(encode_z)
        return ValidationResult(real_image=img, fake_image=fake_img, recon_image=recon_img, encode_latent=encode_z)
