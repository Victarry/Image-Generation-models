## Vector quantized autoregressive model
import torch
from omegaconf import OmegaConf
from torch import nn
from torch import Tensor
import math
from einops import rearrange
from src.models.base import BaseModel, ValidationResult
from src.utils.losses import normal_kld
import torch.nn.functional as F
import numpy as np

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)

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

class TAR(BaseModel):
    def __init__(
        self,
        datamodule: OmegaConf = None,
        lr: float = 1e-4,
        b1: float = 0.9,
        b2: float = 0.999,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()
        self.n_tokens = 3 # 0-255 and <sos>
        self.pos_embed = PositionalEncoding(d_model, H=self.height, W=self.width)
        self.pixel_embed = nn.Embedding(num_embeddings=self.n_tokens, embedding_dim=d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        self.proj = nn.Linear(d_model, self.n_tokens)

    def img2tokens(self, imgs):
        N = imgs.shape[0]
        # imgs = (imgs * 255 + 0.5).long().clamp(0, 255)
        imgs[imgs >= 0.5] = 1
        imgs[imgs < 0.5] = 0
        tokens = rearrange(imgs.long(), 'n c h w -> (h w c) n')
        # prepend <sos> to tokens
        sos = torch.zeros(1, N, device=self.device, dtype=torch.long).fill_(self.n_tokens-1) # start of sequence
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

        tokens = self.img2tokens(imgs)

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
    
    def sample(self, shape, tokens=None):
        if tokens == None:
            N, C, H, W = shape
            tokens = torch.zeros(1+H*W*C, N, device=self.device).long().fill_(-1)
            tokens[0].fill_(self.n_tokens-1) # set <sos> to index 0

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

        tokens = self.img2tokens(imgs)
        loss = self.cal_loss(tokens)

        random_tokens = torch.randint(0, 2, (C*H*W+1, N), device=self.device)
        random_tokens[0] = 0
        rand_loss = self.cal_loss(random_tokens)

        fake_imgs = None
        mask_image = None
        if batch_idx == 0:
            fake_imgs = self.sample(shape).float()

            tokens[H*W*C // 2:] = -1
            mask_image = self.sample(shape, tokens=tokens)

            fake_tokens = self.img2tokens(mask_image)
            fake_loss = self.cal_loss(fake_tokens)
            self.log("var_log/fake_bpg", fake_loss / (H*W*C) / np.log(2))

        self.log("val_log/bpd", loss / (H*W*C) / np.log(2))
        self.log("val_log/rand_bpd", rand_loss / (H*W*C) / np.log(2))

        return ValidationResult(real_image=imgs, fake_image=fake_imgs, others={"mask_image": mask_image})
