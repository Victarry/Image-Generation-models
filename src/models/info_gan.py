import torchvision
import torch
import hydra
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from src.models.base import BaseModel
from src.utils import utils
import itertools


class InfoGAN(BaseModel):
    def __init__(
        self,
        channels,
        width,
        height,
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
        input_normalize=True,
        optim="adam",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = discrete_dim*discrete_value + continuous_dim + noise_dim 
        # networks
        self.netG = hydra.utils.instantiate(netG, input_channel=self.latent_dim, output_channel=channels)
        self.common_layer = hydra.utils.instantiate(netD, input_channel=channels, output_channel=encode_dim)
        self.netD = nn.Sequential(nn.LeakyReLU(), nn.Linear(encode_dim, 1))
        self.netQ = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(encode_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, discrete_dim*discrete_value + continuous_dim),
        )
    
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
        output = output.reshape(
            z.shape[0], self.hparams.channels, self.hparams.height, self.hparams.width
        )
        if return_latents:
            return output, (dis_c_index, cont_c, z)
        else:
            return output

    def adversarial_loss(self, y_hat, y):
        if self.hparams.loss_mode == "vanilla":
            return F.binary_cross_entropy_with_logits(y_hat, y)
        elif self.hparams.loss_mode == "lsgan":
            return F.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        N = imgs.shape[0]

        # train generator
        if optimizer_idx == 0:

            # generate images
            generated_imgs, (dis_c, cont_c, z) = self.decode(N, return_latents=True)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            adv_logit, dis_c_logits, cont_c_hat = self.encode(generated_imgs, return_posterior=True)
            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(adv_logit, valid)
            self.log("train_loss/g_loss", g_loss)

            # mutual information loss
            I_discete_loss = F.cross_entropy(dis_c_logits, dis_c) 
            I_continuous_loss = F.mse_loss(cont_c_hat, cont_c)
            I_loss = I_discete_loss + I_continuous_loss
            self.log("train_loss/I_discrete_loss", I_discete_loss)
            self.log("train_loss/I_continuous", I_continuous_loss)

            # log sampled images
            if self.global_step % 50 == 0:
                self.log_images(generated_imgs, "generated_images", nrow=8)

                N = 4
                a, b, c = self.hparams.discrete_value, self.hparams.continuous_dim, self.hparams.noise_dim
                # each row has `a` values and totally N rows
                # Traverse over discrete latent value while other values are fixed for each N
                disc_c = torch.arange(a).reshape(1, a).repeat(N, 1).reshape(N*a, 1).to(self.device)
                cont_c = torch.randn(N, 1, b).repeat(1, a, 1).reshape(N*a, b).to(self.device) 
                z = torch.randn(N, 1, c).repeat(1, a, 1).reshape(N*a, c).to(self.device) # (40, noise_dim)
                imgs = self.decode(40, disc_c, cont_c, z)
                self.log_images(imgs, "traverse over discrete values", nrow=10)

                # Traverse over continuous latent values while other values are fixed for each N
                disc_c = torch.randperm(a)[:N].reshape(N, 1).repeat(1, a).reshape(N*a, 1).to(self.device)
                cont_c_variation = torch.linspace(-1, 1, a).reshape(1, a).repeat(N, 1).reshape(N*a).to(self.device)
                cont_c = torch.randn(N, 1, b).repeat(1, a, 1).reshape(N*a, b).to(self.device) 
                z = torch.randn(N, 1, c).repeat(1, a, 1).reshape(N*a, c).to(self.device) # (40, noise_dim)

                cont_c_mix = cont_c.clone()
                cont_c_mix[:, 0] = cont_c_variation
                imgs = self.decode(40, disc_c, cont_c_mix, z)
                self.log_images(imgs, "traverse over first continuous values", nrow=10)

                cont_c_mix = cont_c.clone()
                cont_c_mix[:, 1] = cont_c_variation
                imgs = self.decode(40, disc_c, cont_c_mix, z)
                self.log_images(imgs, "traverse over second continuous values", nrow=10)

            return g_loss + self.hparams.lambda_I * I_loss

        # train discriminator
        if optimizer_idx == 1:
            # real loss
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            real_logit = self.netD(self.common_layer(imgs))
            real_loss = self.adversarial_loss(real_logit, valid)

            # fake loss
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)
            fake_logit = self.encode(self.decode(N).detach())
            fake_loss = self.adversarial_loss(fake_logit, fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("train_loss/d_loss", d_loss)
            self.log("train_log/real_logit", real_logit.mean())
            self.log("train_log/fake_logit", fake_logit.mean())

            return d_loss

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

        if self.hparams.optim == "adam":
            opt_g = torch.optim.Adam(
                [{"params": g_param, "lr": lrG}, {"params": q_param, "lr": lrQ}],
                betas=(b1, b2),
            )
            opt_d = torch.optim.Adam(d_param, lr=lrD, betas=(b1, b2))
        elif self.hparams.optim == "sgd":
            opt_g = torch.optim.SGD(g_param, lr=lrG)
            opt_d = torch.optim.SGD(d_param, lr=lrD)
        return [opt_g, opt_d]
