import pytorch_lightning as pl
import torchvision
from src.networks import MLPEncoder, MLPDecoder
import torch
import torch.nn.functional as F

# Tips:
# 1. normalize input and output image range to [-1, 1]

class GAN(pl.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = channels*height*width
        self.generator = MLPDecoder(input_dim=latent_dim, output_dim=data_shape, hidden_dims=[256, 512, 1024])
        self.discriminator = MLPEncoder(input_dim=data_shape, output_dim=1, hidden_dims=[1024, 512, 256])


    def forward(self, z):
        output = self.generator(z)
        output = output.reshape(z.shape[0], self.hparams.channels, self.hparams.height, self.hparams.width)
        return output

    def adversarial_loss(self, y_hat, y):

        # return F.binary_cross_entropy_with_logits(y_hat, y)
        return F.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        imgs = 2*imgs-1

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            generated_imgs = self(z)

            # log sampled images
            sample_imgs = (generated_imgs+1)/2
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, self.global_step)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(generated_imgs), valid)
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
            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log('train_loss/d_loss', d_loss)
            return d_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d]
