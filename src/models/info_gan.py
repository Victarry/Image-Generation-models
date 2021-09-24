import torchvision
import torch
import hydra
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from src.utils import utils
import itertools 


class InfoGAN(pl.LightningModule):
    def __init__(self,
                 channels,
                 width,
                 height,
                 netG,
                 netD,
                 lambda_I=1,
                 latent_dim=100,
                 encode_dim=1024,
                 content_dim=10,
                 loss_mode='vanilla',
                 lrG: float = 0.001,
                 lrD: float = 0.0002,
                 lrQ: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 input_normalize=True,
                 optim='adam',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.netG = hydra.utils.instantiate(netG)
        self.common_layer = hydra.utils.instantiate(netD)
        self.netD = nn.Sequential(nn.LeakyReLU(),
                                  nn.Linear(encode_dim, 1))
        self.netQ = nn.Sequential(nn.LeakyReLU(),
                                  nn.Linear(encode_dim, 128), nn.LeakyReLU(),
                                  nn.Linear(128, content_dim), nn.Softmax())

    def forward(self, N, z=None, c=None):
        if z == None:
            z = torch.randn(N, self.hparams.latent_dim-self.hparams.content_dim).to(self.device)
        if c == None:
            c = torch.zeros(N, self.hparams.content_dim).to(self.device)
            c_index = torch.randint(0, self.hparams.content_dim, (N, )).to(self.device)
            c[torch.arange(N), c_index] = 1 
        
        output = self.netG(torch.cat([z, c], dim=1))
        output = output.reshape(z.shape[0], self.hparams.channels,
                                self.hparams.height, self.hparams.width)
        return output

    def adversarial_loss(self, y_hat, y):
        if self.hparams.loss_mode == 'vanilla':
            return F.binary_cross_entropy_with_logits(y_hat, y)
        elif self.hparams.loss_mode == 'lsgan':
            return F.mse_loss(y_hat, y)

    def log_images(self, imgs, name):
        imgs = imgs.reshape(-1, self.hparams.channels, self.hparams.height, self.hparams.width)
        if self.hparams.input_normalize:
            grid = torchvision.utils.make_grid(imgs[:64],
                                               normalize=True,
                                               value_range=(-1, 1))
        else:
            grid = torchvision.utils.make_grid(imgs[:64], normalize=False)
        self.logger.experiment.add_image(name, grid,
                                         self.global_step)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        N = imgs.shape[0]
        if self.hparams.input_normalize:
            imgs = imgs * 2 - 1

        # sample noise
        z = torch.randn(N, self.hparams.latent_dim-self.hparams.content_dim).to(self.device)
        c = torch.zeros(N, self.hparams.content_dim).to(self.device)
        c_index = torch.randint(0, self.hparams.content_dim, (N, )).to(self.device)
        c[torch.arange(N), c_index] = 1
        input_latent = torch.cat([z, c], dim=1)

        # train generator
        if optimizer_idx == 0:

            # generate images
            generated_imgs = self.netG(input_latent)

            # log sampled images
            if self.global_step % 50 == 0:
                self.log_images(generated_imgs, 'generated_images')

                index = torch.arange(10).reshape(10, 1).repeat(1, 8).reshape(80)  # (80)
                sample_c = torch.zeros(80, 10).to(self.device)
                sample_c[torch.arange(80), index] = 1

                z = torch.randn(80, self.hparams.latent_dim-self.hparams.content_dim).to(self.device)
                sample_image = self.netG(torch.cat([z, sample_c], dim=1))
                self.log_images(sample_image, 'static_images')

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            common_feature = self.common_layer(generated_imgs)
            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.netD(common_feature), valid)
            self.log('train_loss/g_loss', g_loss)

            # mutual information loss
            c_hat = self.netQ(common_feature)
            I_loss = F.cross_entropy(c_hat, c_index)
            self.log('train_loss/I_loss', I_loss)

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
            fake_logit = self.netD(self.common_layer(self.netG(input_latent).detach()))
            fake_loss = self.adversarial_loss(fake_logit, fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log('train_loss/d_loss', d_loss)
            self.log('train_log/real_logit', real_logit.mean())
            self.log('train_log/fake_logit', fake_logit.mean())

            return d_loss

    def configure_optimizers(self):
        lrG = self.hparams.lrG
        lrD = self.hparams.lrD
        lrQ = self.hparams.lrQ
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        q_param = self.netQ.parameters()
        g_param = self.netG.parameters()
        d_param = itertools.chain(self.netD.parameters(), self.common_layer.parameters())

        if self.hparams.optim == 'adam':
            opt_g = torch.optim.Adam([{'params': g_param, 'lr': lrG}, {'params': q_param, 'lr': lrQ}], betas=(b1, b2))
            opt_d = torch.optim.Adam(d_param, lr=lrD, betas=(b1, b2))
        elif self.hparams.optim == 'sgd':
            opt_g = torch.optim.SGD(g_param, lr=lrG)
            opt_d = torch.optim.SGD(d_param, lr=lrD)
        return [opt_g, opt_d]
