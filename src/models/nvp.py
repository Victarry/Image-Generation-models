import torch
from torch import nn
from einops import rearrange, reduce
import torch.nn.functional as F
from torch import optim

from src.models.base import BaseModel, ValidationResult


class ResBlock(nn.Module):
    def __init__(self, hidden_dim=32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return F.relu(x) + res


class CoupleLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, mask_type=0, mask_id=0):
        super().__init__()
        self.mask_type = mask_type  # 0: checkboard 1: channels
        self.mask_id = mask_id

        self.scale = nn.Parameter(data=torch.ones(1))
        self.net_s = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 3, padding=1), 
                            *[ResBlock(hidden_dim) for _ in range(4)], 
                            nn.Conv2d(hidden_dim, in_dim, 1),
                            nn.Tanh()
                            )

        self.net_t = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 3, padding=1), 
                            *[ResBlock(hidden_dim) for _ in range(4)], 
                            nn.Conv2d(hidden_dim, in_dim, 1))

    def get_mask(self, x):
        mask = torch.zeros_like(x)
        if self.mask_type == 0:
            mask = rearrange(mask, "n c (h h2) (w w2) -> n c h h2 w w2", h2=2, w2=2)
            if self.mask_id % 2 == 0:
                mask[:, :, :, 0, :, 0] = 1
                mask[:, :, :, 1, :, 1] = 1
            else:
                mask[:, :, :, 0, :, 0] = 1
                mask[:, :, :, 1, :, 1] = 1
            mask = rearrange(mask, "n c h h2 w w2 -> n c (h h2) (w w2)")
        elif self.mask_type == 1:
            n, c, h, w = x.shape
            half_channel = c // 2
            if self.mask_id % 2 == 0:
                mask[:, :half_channel] = 1
            else:
                mask[:, half_channel:] = 1
        return mask

    def forward(self, x):
        mask = self.get_mask(x)
        mask_x = mask * x
        scale = torch.exp(self.scale * self.net_s(mask_x))
        bias = self.net_t(mask_x)
        y = mask_x + (1 - mask) * (x * scale + bias)
        return y, reduce((1 - mask) * scale, "n c h w -> n", reduction="sum")

    def reverse(self, y):
        mask = self.get_mask(y)
        mask_x = y * mask
        inv_scale = torch.exp(-self.scale * self.net_s(mask_x))
        bias = self.net_t(mask_x)

        x = (1 - mask) * (y - bias) * inv_scale + mask_x
        return x

class CoupleBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, mask_type=0) -> None:
        super().__init__()
        self.layer1 = CoupleLayer(in_dim, hidden_dim, mask_type, 0)
        self.layer2 = CoupleLayer(in_dim, hidden_dim, mask_type, 1)
        self.layer3 = CoupleLayer(in_dim, hidden_dim, mask_type, 2)
    
    def forward(self, x):
        x, det1 = self.layer1(x)
        x, det2 = self.layer2(x)
        x, det3 = self.layer3(x)

        return x, det1 + det2 + det3

    def reverse(self, x):
        x = self.layer3.reverse(x)
        x = self.layer2.reverse(x)
        x = self.layer1.reverse(x)
        return x

def squeeze(x):
    x = rearrange(x, "n c (h k1) (w k2) -> n (c k1 k2) h w", k1=2, k2=2)
    return x

def unsqueeze(x):
    x = rearrange(x, "n (c k1 k2) h w -> n c (h k1) (w k2)", k1=2, k2=2)
    return x

class RealNVPNetwork(nn.Module):
    def __init__(self, in_channel=1, hidden_dim=32):
        super().__init__()
        self.block1 = CoupleBlock(in_channel, hidden_dim, 0) # checkboard

        self.block2_1 = CoupleBlock(in_channel*4, hidden_dim, 1) # channel
        self.block2_2 = CoupleBlock(in_channel*4, hidden_dim, 1) # checkboard

        self.block3_1 = CoupleBlock(in_channel*16, hidden_dim, 1) # channel
        self.block3_2 = CoupleBlock(in_channel*16, hidden_dim, 1) # checkboard

    def forward(self, x):
        dets = {}
        x, dets["1"] = self.block1(x)
        x = squeeze(x)

        x, dets["2_1"] = self.block2_1(x)
        x, dets["2_2"] = self.block2_2(x)
        x = squeeze(x)


        x, dets["3_1"] = self.block3_1(x)
        x, dets["3_2"] = self.block3_2(x)
        return x, sum(det for det in dets.values())

    def reverse(self, x):
        x = self.block3_2.reverse(x)
        x = self.block3_1.reverse(x)
        x = unsqueeze(x)

        x = self.block2_2.reverse(x)
        x = self.block2_1.reverse(x)
        x = unsqueeze(x)
        
        x = self.block1.reverse(x)
        return x

class RealNVP(BaseModel):
    def __init__(
        self, 
        datamodule, 
        hidden_dim, 
        lr=1e-3
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()
        self.model = RealNVPNetwork(self.channels, hidden_dim)

        self.latent_shape = [self.channels*16, self.height // 4, self.width // 4]
        self.register_buffer("log2", torch.log(torch.tensor(2, dtype=torch.float32, device=self.device)))

    def forward(self, x):
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor in range (0, 1).
        """
        z, det = self.model(x)
        return z, det
    
    def sample(self, N):
        latents = torch.randn(N, *self.latent_shape, device=self.device)
        imgs = self.model.reverse(latents)
        return imgs

    def calc_likelihood(self, x):
        z, det = self.forward(x)
        dist = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        log_prob = dist.log_prob(z).sum(dim=[1,2,3]) + torch.log(det)
        return -log_prob.mean(dim=0) / self.log2

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        loss = self.calc_likelihood(img)
        self.log("train_bpd", loss) # bpd: bits per dim, by entropy encoding
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        N, C, H, W = img.shape
        loss = self.calc_likelihood(img)
        self.log("val_bpd", loss)

        sample_img = None
        if batch_idx == 0:
            sample_img = self.sample(N)
        return ValidationResult(real_image=img, fake_image=sample_img)
