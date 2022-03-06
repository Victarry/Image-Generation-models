import torch
import hydra
import torch.nn.functional as F
from torch import nn
from src.models.base import BaseModel
from src.utils.losses import adversarial_loss
from src.callbacks.visualization import get_grid_images
import itertools


class InfoGAN(BaseModel):
    def __init__(
        self,
        datamodule,
        netG,
        netD,
        lambda_I=1, # loss weight for mutual information
        discrete_dim=1, # discrete latent variable dimension
        discrete_value=10, # the value range of discrete latent variable
        continuous_dim=2,
        noise_dim=62,
        encode_dim=1024, # intermediate dim for common layer
        loss_mode="vanilla",
        lrG: float = 0.001,
        lrD: float = 0.0002,
        lrQ: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()

        self.latent_dim = discrete_dim*discrete_value + continuous_dim + noise_dim 
        # networks
        self.netG = hydra.utils.instantiate(netG, input_channel=self.latent_dim, output_channel=self.channels)
        self.common_layer = hydra.utils.instantiate(netD, input_channel=self.channels, output_channel=encode_dim)
        self.netD = nn.Sequential(nn.LeakyReLU(), nn.Linear(encode_dim, 1))
        self.netQ = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(encode_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, discrete_dim*discrete_value + continuous_dim),
        )

    def configure_optimizers(self):
        lrG = self.hparams.lrG
        lrD = self.hparams.lrD
        lrQ = self.hparams.lrQ
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        q_param = self.netQ.parameters()
        g_param = self.netG.parameters()
        d_param = itertools.chain(
            self.netD.parameters(), self.common_layer.parameters()
        )

        opt_g = torch.optim.Adam(
            [{"params": g_param, "lr": lrG}, {"params": q_param, "lr": lrQ}],
            betas=(b1, b2),
        )
        opt_d = torch.optim.Adam(d_param, lr=lrD, betas=(b1, b2))
        return [opt_g, opt_d]
    
    def encode(self, x, return_posterior=False):
        x = self.common_layer(x)
        adv_logit = self.netD(x)
        if return_posterior:
            output = self.netQ(x)
            dis_c_logits = output[:, :-self.hparams.continuous_dim].reshape(-1, 
                self.hparams.discrete_value, self.hparams.discrete_dim)
            cont_c = output[:, -self.hparams.continuous_dim:]
            return adv_logit, dis_c_logits, cont_c
        else:
            return adv_logit

    def decode(self, N, dis_c_index=None, cont_c=None, z=None, return_latents=False):
        """
        N: batch_size
        disc_c_index: tensor of shape (N, discrete_dim)
        """
        if dis_c_index == None:
            dis_c_index = torch.randint(0, self.hparams.discrete_value, (N, self.hparams.discrete_dim)).to(self.device) # (N, discrete_dim)
        dis_c = torch.zeros(N, self.hparams.discrete_value, self.hparams.discrete_dim).to(self.device) # (N, discrete_value, disrete_dim)
        dis_c.scatter_(1, dis_c_index.unsqueeze(1), torch.ones_like(dis_c))
        
        if cont_c == None:
            cont_c = torch.zeros(N, self.hparams.continuous_dim, device=self.device).uniform_(-1, 1)

        if z == None:
            z = torch.randn(N, self.hparams.noise_dim).to(self.device)

        output = self.netG(torch.cat([dis_c.reshape(N, -1), cont_c, z], dim=1))
        output = output.reshape(z.shape[0], self.channels, self.height, self.width)
        if return_latents:
            return output, (dis_c_index, cont_c, z)
        else:
            return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        N = imgs.shape[0]

        # train generator
        if optimizer_idx == 0:
            generated_imgs, (dis_c, cont_c, z) = self.decode(N, return_latents=True)
            adv_logit, dis_c_logits, cont_c_hat = self.encode(generated_imgs, return_posterior=True)
            g_loss = adversarial_loss(adv_logit, target_is_real=True)

            # mutual information loss
            I_discete_loss = F.cross_entropy(dis_c_logits, dis_c) 
            I_continuous_loss = F.mse_loss(cont_c_hat, cont_c)
            I_loss = I_discete_loss + I_continuous_loss

            self.log("train_loss/g_loss", g_loss)
            self.log("train_loss/I_discrete_loss", I_discete_loss)
            self.log("train_loss/I_continuous", I_continuous_loss)

            return g_loss + self.hparams.lambda_I * I_loss

        if optimizer_idx == 1:
            pred_real = self.netD(self.common_layer(imgs))
            real_loss = adversarial_loss(pred_real, target_is_real=True)

            pred_fake = self.encode(self.decode(N).detach())
            fake_loss = adversarial_loss(pred_fake, target_is_real=False)

            d_loss = (real_loss + fake_loss) / 2

            self.log("train_loss/d_loss", d_loss)
            self.log("train_log/pred_real", pred_real.mean())
            self.log("train_log/pred_fake", pred_fake.mean())

            return d_loss

    def on_train_epoch_end(self) -> None:
        generated_images = self.decode(64)
        grid_images = get_grid_images(generated_images, self, 64, 8)
        self.logger.experiment.add_image("images/sample", grid_images, global_step=self.current_epoch)

        N = 8 # row of images
        a, b, c = self.hparams.discrete_value, self.hparams.continuous_dim, self.hparams.noise_dim
        # each row has `a` values and totally N rows
        # Traverse over discrete latent value while other values are fixed for each N
        disc_c = torch.arange(a).reshape(1, a).repeat(N, 1).reshape(N*a, 1).to(self.device)
        cont_c = torch.randn(N, 1, b).repeat(1, a, 1).reshape(N*a, b).to(self.device) 
        z = torch.randn(N, 1, c).repeat(1, a, 1).reshape(N*a, c).to(self.device) # (40, noise_dim)
        imgs = self.decode(N*a, disc_c, cont_c, z)

        grid_images = get_grid_images(imgs, self, N*a, a)
        self.logger.experiment.add_image("visual/traverse over discrete values", grid_images, global_step=self.current_epoch)

        col = 10
        # Traverse over continuous latent values while other values are fixed for each N
        disc_c = torch.randint(low=0, high=a, size=(N, 1)).repeat(1, col).reshape(N*col, 1).to(self.device)
        cont_c_variation = torch.linspace(-2, 2, col).reshape(1, col).repeat(N, 1).reshape(N*col).to(self.device)
        cont_c = torch.randn(N, 1, b).repeat(1, col, 1).reshape(N*col, b).to(self.device) 
        z = torch.randn(N, 1, c).repeat(1, col, 1).reshape(N*col, c).to(self.device) # (N*a, noise_dim)

        cont_c_mix = cont_c.clone()
        cont_c_mix[:, 0] = cont_c_variation
        imgs = self.decode(N*col, disc_c, cont_c_mix, z)
        grid_images = get_grid_images(imgs, self, N*col, col)
        self.logger.experiment.add_image("visual/traverse over first continuous values", grid_images, global_step=self.current_epoch)

        cont_c_mix = cont_c.clone()
        cont_c_mix[:, 1] = cont_c_variation
        imgs = self.decode(N*col, disc_c, cont_c_mix, z)
        grid_images = get_grid_images(imgs, self, N*col, col)
        self.logger.experiment.add_image("visual/traverse over second continuous values", grid_images, global_step=self.current_epoch)