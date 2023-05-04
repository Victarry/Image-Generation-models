import hydra
import torch
from .base import BaseModel, ValidationResult


class WGAN(BaseModel):
    """Wassertain GAN. https://arxiv.org/abs/1701.07875
    Training tricks:
        1. As paper said, momentum based optimizer like Adam performs worse, so we use RMSProp here.
    """
    def __init__(
        self,
        datamodule,
        netG,
        netD,
        latent_dim=100,
        n_critic=5,
        clip_weight=0.01,
        lrG: float = 5e-5,
        lrD: float = 5e-5,
        alpha: float = 0.99,
        eval_fid=False,
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        self.generator = hydra.utils.instantiate(
            netG, input_channel=latent_dim, output_channel=self.channels
        )
        self.console.info(f"Generator architecture: \n {self.generator}")
        self.discriminator = hydra.utils.instantiate(
            netD, input_channel=self.channels, output_channel=1
        )
        self.console.info(f"Discriminator architecture: \n {self.discriminator}")

    def forward(self, z):
        output = self.generator(z)
        output = output.reshape(
            z.shape[0], self.channels, self.height, self.width
        )
        return output

    def configure_optimizers(self):
        lrG = self.hparams.lrG
        lrD = self.hparams.lrD
        alpha = self.hparams.alpha

        opt_g = torch.optim.RMSprop(
            self.generator.parameters(), lr=lrG, alpha=alpha
        )
        opt_d = torch.optim.RMSprop(
            self.discriminator.parameters(), lr=lrD, alpha=alpha
        )
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):
        imgs, _ = batch  # (N, C, H, W)

        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)  # (N, latent_dim)
        z = z.type_as(imgs)

        opt_g, opt_d = self.optimizers()

        # clip discriminator weight for 1-Lipschitz constraint
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.hparams.clip_weight, self.hparams.clip_weight)

        if batch_idx % (self.hparams.n_critic+1) == 0:
            generated_imgs = self(z)
            g_loss = -torch.mean(self.discriminator(generated_imgs))

            opt_g.zero_grad()
            self.manual_backward(g_loss)
            opt_g.step()

            self.log("train_loss/g_loss", g_loss, prog_bar=True)

        else:
            real_loss = -self.discriminator(imgs).mean()
            fake_loss = self.discriminator(self(z).detach()).mean()
            d_loss = real_loss + fake_loss
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()

            self.log("train_loss/d_loss", d_loss)
            self.log("train_log/real_logit", -real_loss)
            self.log("train_log/fake_logit", fake_loss)


    def validation_step(self, batch, batch_idx):
        img, _ = batch
        z = torch.randn(img.shape[0], self.hparams.latent_dim).to(self.device)
        fake_imgs = self.forward(z)
        return ValidationResult(real_image=img, fake_image=fake_imgs)
