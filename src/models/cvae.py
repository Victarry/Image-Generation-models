from hydra.utils import instantiate
import torch
from omegaconf import OmegaConf

from src.models.base import BaseModel, ValidationResult
from torch import distributions
from src.utils.distributions import get_decode_dist
from src.utils.losses import normal_kld
import torch.nn.functional as F

## q(z | x, c): encoder, append one-hot vector to all pixels
## p(x | z, c): decoder, append embedding to z
## p(z | c): prior distribution, can be fixed or learned
## Loss: -KL(q(z|x,c) || p(z|c)) + E_{z~q(z|x,c)}(log p(x|z, c))

class cVAE(BaseModel):
    def __init__(
        self,
        datamodule: OmegaConf = None,
        encoder: OmegaConf = None,
        decoder: OmegaConf = None,
        latent_dim: int = 100,
        beta: float = 1.0,
        recon_weight: float = 1.0,
        lr: float = 1e-4,
        b1: float = 0.9,
        b2: float = 0.999,
        n_classes: int = None,
        encode_label: bool = True,
        decoder_dist = "guassian"
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()

        self.decoder = instantiate(decoder, input_channel=latent_dim*2, output_channel=self.channels, output_act=self.output_act)
        if encode_label:
            self.encoder = instantiate(encoder, input_channel=self.channels+n_classes, output_channel=2 * latent_dim)
        else:
            self.encoder = instantiate(encoder, input_channel=self.channels, output_channel=2 * latent_dim)
        self.class_embedding = torch.nn.Embedding(n_classes, latent_dim)
        self.decoder_dist = get_decode_dist(decoder_dist)
        self.n_classes = n_classes

    def forward(self, z, labels):
        """Generate images given latent code."""
        embed = self.class_embedding(labels)
        z = torch.cat([z, embed], dim=1)
        output = self.decoder(z)
        output = output.reshape(output.shape[0], self.channels, self.height, self.width)
        return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=(b1, b2))
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.99)
        return [opt], [scheduler]
    
    def reparameterize(self, mu, log_sigma):
        post_dist = distributions.Normal(mu, torch.exp(log_sigma))
        samples_z = post_dist.rsample()
        return samples_z

    def vae(self, imgs, labels):
        N, C, H, W = imgs.shape
        if self.hparams.encode_label:
            labels_onehot = F.one_hot(labels, num_classes=self.n_classes).reshape(N, self.n_classes, 1, 1).expand(N, self.n_classes, H, W)
            imgs = torch.cat([imgs, labels_onehot], dim=1)
        z_ = self.encoder(imgs)  # (N, latent_dim)
        mu, log_sigma = torch.chunk(z_, chunks=2, dim=1)
        z = self.reparameterize(mu, log_sigma)
        recon_imgs = self.forward(z, labels)
        return mu, log_sigma, z, recon_imgs

    def training_step(self, batch, batch_idx):
        imgs, labels = batch # (N, C, H, W)
        mu, log_sigma, z, recon_imgs = self.vae(imgs, labels)
        kld = normal_kld(mu, log_sigma)

        log_p_x_of_z = self.decoder_dist.prob(recon_imgs, imgs).mean(dim=0)
        elbo = -self.hparams.beta*kld + self.hparams.recon_weight * log_p_x_of_z

        self.log("train_log/elbo", elbo)
        self.log("train_log/kl_divergence", kld)
        self.log("train_log/log_p_x_of_z", log_p_x_of_z)
        return -elbo 
    
    def sample(self, N):
        labels = torch.arange(self.n_classes, device=self.device).reshape(self.n_classes, 1).repeat(1, N).reshape(-1)
        z = torch.randn(N*self.n_classes, self.hparams.latent_dim, device=self.device)
        imgs = self.forward(z, labels)
        return imgs
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        mu, log_sigma, z, recon_imgs = self.vae(imgs, labels)
        log_p_x_of_z = self.decoder_dist.prob(recon_imgs, imgs).mean(dim=0)

        fake_imgs = self.sample(8)
        self.log("val_log/log_p_x_of_z", log_p_x_of_z)
        return ValidationResult(real_image=imgs, fake_image=fake_imgs, 
                    recon_image=recon_imgs, label=labels, encode_latent=z)