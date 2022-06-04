## Vector quantized autoregressive model
import torch
from omegaconf import OmegaConf
from torch import embedding, embedding_renorm_, nn
from torch import Tensor
import math
from einops import rearrange
from src.models.base import BaseModel, ValidationResult
from src.utils.losses import normal_kld
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, H: int, W: int) -> None:
        super().__init__()
        self.h_pe = torch.nn.Parameter(data=torch.randn(H, 1, d_model))
        self.w_pe = torch.nn.Parameter(data=torch.randn(W, 1, d_model))
        self.first_pe = torch.nn.Parameter(data=torch.randn(1, 1, d_model))
        self.H = H
        self.W = W
    
    def forward(self, x):
        h_pe = self.h_pe.repeat(1, self.W, 1).reshape(self.H*self.W, 1, -1)
        h_pe = torch.cat([self.first_pe, h_pe], dim=0)

        w_pe = self.w_pe.repeat(self.H, 1, 1).reshape(self.H*self.W, 1, -1)
        w_pe = torch.cat([self.first_pe, w_pe], dim=0)

        x = x + h_pe[:x.size(0)] + w_pe[:x.size(0)]
        return x

class PixelEncoding(nn.Module):
    def __init__(self, n_tokens, d_model, class_cond=False, n_classes=None) -> None:
        super().__init__()
        self.pixel_embed = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
        if class_cond:
            self.cond_embed = nn.Embedding(num_embeddings=n_classes, embedding_dim=d_model)
        else:
            self.cond_embed = nn.Embedding(num_embeddings=1, embedding_dim=d_model)
        
    def forward(self, tokens):
        token1 = self.cond_embed(tokens[0:1])
        token2 = self.pixel_embed(tokens[1:])
        return torch.cat([token1, token2], dim=0)


class TAR(BaseModel):
    def __init__(
        self,
        datamodule: OmegaConf = None,
        lr: float = 1e-4,
        b1: float = 0.9,
        b2: float = 0.999,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        class_cond: bool = False,
        n_classes: int = 10
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()
        self.n_tokens = 2 # 0-255 and <sos>

        self.pos_embed = PositionalEncoding(d_model, H=self.height, W=self.width)
        self.pixel_embed = PixelEncoding(self.n_tokens, d_model, class_cond, n_classes=n_classes)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, self.n_tokens)

    def img2tokens(self, imgs, label):
        N = imgs.shape[0]
        # imgs = (imgs * 255 + 0.5).long().clamp(0, 255)
        imgs[imgs >= 0.5] = 1
        imgs[imgs < 0.5] = 0
        tokens = rearrange(imgs.long(), 'n c h w -> (h w c) n')
        # prepend <sos> to tokens
        if self.hparams.class_cond:
            sos = label.long().reshape(1, N) # start of sequence
        else:
            sos = torch.zeros(1, N, device=self.device, dtype=torch.long) # start of sequence
        tokens = torch.cat([sos, tokens], dim=0) # (seq_len+1, batch)
        return tokens

    def tokens2img(self, tokens, shape):
        N, C, H, W = shape
        imgs = rearrange(tokens[1:], "(h w c) n -> n c h w", n=N, c=C, h=H, w=W).float()
        return imgs

    def forward(self, tokens):
        # tokens: (seq_len, batch) consisting of long tensor
        # returns: (seq_len, batch, n_classes)
        S, N = tokens.shape
        mask = torch.tril(torch.ones(S, S, device=self.device)) == 0 # True indicates not to attend!
        # append start of sentence token
        embed = self.pos_embed(self.pixel_embed(tokens))  # s n d
        features = self.encoder.forward(embed, mask=mask) # s n d
        pred = self.proj(features) # s n num_class
        pred = rearrange(pred, "s n c -> s c n")
        return pred

    def cal_loss(self, tokens):
        # tokens: (S+1, N), including <sos> token
        pred = self.forward(tokens) # (S+1, n_class, N)
        loss = F.cross_entropy(pred[:-1], tokens[1:], reduction="none").sum(dim=0).mean()
        return loss

    def training_step(self, batch, batch_idx):
        imgs, labels = batch # (N, C, H, W)
        N, C, H, W = imgs.shape

        tokens = self.img2tokens(imgs, labels)

        loss = self.cal_loss(tokens)
        self.log("train_log/nll", loss)
        self.log("train_log/bpd", loss / (H*W*C) / np.log(2))
        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=(b1, b2))
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.99)
        return [opt], [scheduler]
    
    def sample(self, shape, tokens=None, labels=None):
        if tokens == None:
            N, C, H, W = shape
            tokens = torch.zeros(1+H*W*C, N, device=self.device).long().fill_(-1)
            if self.hparams.class_cond:
                tokens[0] = labels
            else:
                tokens[0].fill_(0) # set <sos> to index 0

        for i in range(tokens.shape[0]-1):
            if (tokens[i+1, :] != -1).all().item():
                continue
            pred = self.forward(tokens[:i+1]) # (S, n_class, N)
            prob = torch.softmax(pred[-1].T, dim=-1)
            sample = torch.multinomial(prob, num_samples=1).squeeze(-1) # (N)
            tokens[i+1] = sample
        imgs = self.tokens2img(tokens, shape)
        return imgs
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        N, C, H, W = shape = imgs.shape

        tokens = self.img2tokens(imgs, labels)
        loss = self.cal_loss(tokens)

        random_tokens = torch.randint(0, 2, (C*H*W+1, N), device=self.device)
        random_tokens[0] = 0
        rand_loss = self.cal_loss(random_tokens)

        fake_imgs = None
        mask_image = None
        if batch_idx == 0:
            fake_labels = None
            if self.hparams.class_cond:
                fake_labels = torch.arange(0, self.hparams.n_classes).reshape(-1, 1).repeat(1, 8).reshape(-1)
            fake_imgs = self.sample((self.hparams.n_classes*8, C, H, W), labels=fake_labels).float()

            tokens[H*W*C // 2:] = -1
            mask_image = self.sample(shape, tokens=tokens)

            fake_tokens = self.img2tokens(mask_image, labels)
            fake_loss = self.cal_loss(fake_tokens)
            self.log("var_log/fake_bpg", fake_loss / (H*W*C) / np.log(2))

        self.log("val_log/bpd", loss / (H*W*C) / np.log(2))
        self.log("val_log/rand_bpd", rand_loss / (H*W*C) / np.log(2))

        return ValidationResult(real_image=imgs, fake_image=fake_imgs, others={"mask_image": mask_image})
