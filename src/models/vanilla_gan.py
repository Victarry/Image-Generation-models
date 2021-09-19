import pytorch_lightning as pl
import torchvision
from src.networks import MLPEncoder, MLPDecoder
import torch
import torch.nn.functional as F


class GAN(pl.LightningModule):
    def __init__(self,
                 channels,
                 width,
                 height,
                 netG,
                 netD,
                 latent_dim=100,
                 loss_mode='vanilla',
                 lrG: float = 0.0002,
                 lrD: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 input_normalize=True,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = channels*width*height
        
        self.generator = MLPDecoder(**netG, output_dim=data_shape)
        self.discriminator = MLPEncoder(input_dim=data_shape, **netD)

    def forward(self, z):
        output = self.generator(z)
        output = output.reshape(z.shape[0], self.hparams.channels,
                                self.hparams.height, self.hparams.width)
        return output

    def adversarial_loss(self, y_hat, y):
        if self.hparams.loss_mode == 'vanilla':
            return F.binary_cross_entropy_with_logits(y_hat, y)
        elif self.hparams.loss_mode == 'lsgan':
            return F.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        if self.hparams.input_normalize:
            imgs = imgs*2-1

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            generated_imgs = self(z)

            # log sampled images
            if self.hparams.input_normalize:
                grid = torchvision.utils.make_grid(generated_imgs, normalize=True, value_range=(-1, 1))
            else:
                grid = torchvision.utils.make_grid(generated_imgs, normalize=False)
            self.logger.experiment.add_image("generated_images", grid,
                                             self.global_step)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(generated_imgs),
                                           valid)
            self.log('train_loss/g_loss', g_loss, prog_bar=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # real loss
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # fake loss
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)
            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log('train_loss/d_loss', d_loss)
            return d_loss

    def configure_optimizers(self):
        lrG = self.hparams.lrG
        lrD = self.hparams.lrD
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=lrG,
                                 betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=lrD,
                                 betas=(b1, b2))
        return [opt_g, opt_d]
