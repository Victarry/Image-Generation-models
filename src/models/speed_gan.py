import torch
from omegaconf import OmegaConf
from .base import BaseModel, ValidationResult
from hydra.utils import instantiate
from src.utils.losses import adversarial_loss

class GAN(BaseModel):
    def __init__(
        self,
        datamodule: OmegaConf,
        netG: OmegaConf,
        netD: OmegaConf,
        latent_dim: int = 100,
        loss_mode: str = "vanilla",
        lrG: float = 0.0002,
        lrD: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()
        self.netG = instantiate(netG, input_channel=latent_dim, output_channel=self.channels)
        self.netD = instantiate(netD, input_channel=self.channels, output_channel=1)
        self.automatic_optimization = False

    def forward(self, z):
        output = self.netG(z)
        output = output.reshape(z.shape[0], self.channels, self.height, self.width)
        return output

    def configure_optimizers(self):
        lrG, lrD = self.hparams.lrG, self.hparams.lrD
        b1, b2 = self.hparams.b1, self.hparams.b2
        opt_g = torch.optim.Adam(self.netG.parameters(), lr=lrG, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.netD.parameters(), lr=lrD, betas=(b1, b2))
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx, optimizer_idx):
        # speed gan
        # generator: forward 1 times, backward 1 times
        # discriminator: forward 2 times, backward 2 times

        # original gan
        # generator: forward 2 times, backward 1 times
        # discriminator: forward 3 times, backward 3 times
        opt_g, opt_d = self.optimizers()
        imgs, _ = batch  # (N, C, H, W)
        N, C, H, W = imgs.shape
        z = torch.randn(N, self.hparams.latent_dim).to(self.device)

        fake_imgs = self.netG(z)
        pred_fake = self.netD(fake_imgs)
        pred_real = self.netD(imgs)

        real_loss = adversarial_loss(pred_real, True, loss_mode=self.hparams.loss_mode)
        fake_loss = adversarial_loss(pred_fake, False, loss_mode=self.hparams.loss_mode)

        g_loss = adversarial_loss(pred_fake, True, loss_mode=self.hparams.loss_mode)
        d_loss = (real_loss + fake_loss) / 2

        opt_g.zero_grad()
        self.manual_backward(g_loss, retain_graph=True)
        opt_g.step()

        opt_d.zero_grad()
        self.manual_backward(d_loss, inputs=list(self.netD.parameters()), retain_graph=True)
        opt_d.step()

        self.log("train_loss/d_loss", d_loss)
        self.log("train_loss/g_loss", g_loss)
        self.log("train_log/pred_real", pred_real.mean())
        self.log("train_log/pred_fake", pred_fake.mean())


    def validation_step(self, batch, batch_idx):
        img, _ = batch
        z = torch.randn(img.shape[0], self.hparams.latent_dim).to(self.device)
        fake_imgs = self.forward(z)
        return ValidationResult(real_image=img, fake_image=fake_imgs)
