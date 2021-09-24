import hydra
import pytorch_lightning as pl
import torchvision
import torch
import torch.nn.functional as F
from src.utils import utils
import itertools


class VAE(pl.LightningModule):
    def __init__(self,
                 channels,
                 width,
                 height,
                 encoder,
                 decoder,
                 reg_weight, 
                 latent_dim=100,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 input_normalize=True,
                 optim='adam',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = channels*width*height
        
        self.decoder = hydra.utils.instantiate(decoder)
        self.encoder = hydra.utils.instantiate(encoder)

        # model info
        self.console = utils.get_logger()
    def forward(self):
        noise = torch.randn(64, self.hparams.latent_dim).to(self.device)

        # decoding
        output = self.decoder(noise)
        output = output.reshape(output.shape[0], self.hparams.channels,
                                self.hparams.height, self.hparams.width)
        return output

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        N = imgs.shape[0]
        if self.hparams.input_normalize:
            imgs = imgs*2-1

        # encoding
        z = self.encoder(imgs) # (N, latent_dim)
        mu, log_sigma = z[:, :self.hparams.latent_dim], z[:, self.hparams.latent_dim:]

        # note the negative mark
        reg_loss = -0.5*torch.sum(1 + 2*log_sigma - mu**2 - torch.exp(2*log_sigma)) / N 

        # reparameterization
        noise = torch.randn(N, self.hparams.latent_dim).type_as(imgs)
        # numerical problem
        samples = noise*torch.exp(log_sigma) + mu

        # decoding
        generated_imgs = self.decoder(samples).reshape(-1, self.hparams.channels, self.hparams.height, self.hparams.width)
        recon_loss = F.mse_loss(generated_imgs, imgs, reduction='sum') / N

        total_loss = self.hparams.reg_weight*reg_loss + recon_loss

        self.log('train_loss/reg_loss', reg_loss.item())
        self.log('train_loss/recon_loss', recon_loss.item())

        # log sampled images
        if self.global_step % 50 == 0:
            sample_images = self()
            if self.hparams.input_normalize:
                input_img = torchvision.utils.make_grid(imgs[:64], normalize=True, value_range=(-1, 1))
                output_img = torchvision.utils.make_grid(generated_imgs[:64], normalize=True, value_range=(-1, 1))
                sample_image = torchvision.utils.make_grid(sample_images[:64], normalize=True, value_range=(-1, 1))
            else:
                input_img = torchvision.utils.make_grid(imgs[:64], normalize=False)
                output_img = torchvision.utils.make_grid(generated_imgs[:64], normalize=False)
                sample_image = torchvision.utils.make_grid(sample_images[:64], normalize=False)
            self.logger.experiment.add_image("recon/source_image", input_img, self.global_step)
            self.logger.experiment.add_image("recon/output_image", output_img, self.global_step)
            self.logger.experiment.add_image("sample/output_image", sample_image, self.global_step)

        return total_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
                                    lr=lr,
                                    betas=(b1, b2))
        return opt 
