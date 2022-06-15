from functools import partial
from types import new_class
from torch import nn
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from src.models.base import BaseModel, ValidationResult
from einops import rearrange


class MaskedLinear(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.model = nn.Linear(in_channel, out_channel)
        self.register_buffer("mask", torch.ones(out_channel, in_channel))
    
    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        return F.linear(x, weight=self.model.weight * self.mask, bias=self.model.bias)

class MADENet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_class, n_layer):
        """
        in_dim: vector length of input data
        hidden_dim: hidden_dim of vectors
        """
        super().__init__()
        self.in_dim = in_dim
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.n_class = n_class

        dims = [in_dim] + [hidden_dim] * n_layer + [in_dim*n_class]
        self.layers = []
        for in_feature, out_feature in zip(dims[:-1], dims[1:]):
            self.layers.append(MaskedLinear(in_feature, out_feature))
        self.model = nn.Sequential(*self.layers)
        self.reset_mask()

    def reset_mask(self):
        low = 0
        high = self.in_dim
        units = []
        data_unit = torch.arange(0, high)
        # data_unit = torch.randperm(high)
        units.append(data_unit)

        for _ in range(self.n_layer):
            hidden_unit = torch.randint(low=low, high=high, size=(self.hidden_dim, ))
            units.append(hidden_unit)
            low = min(hidden_unit)
        units.append(data_unit.unsqueeze(1).repeat(1, self.n_class).reshape(-1) - 1)

        for layer, in_unit, out_unit in zip(self.layers, units[:-1], units[1:]):
            mask = out_unit.unsqueeze(1) >= in_unit # (out_features, int_features)
            layer.set_mask(mask)

    def forward(self, x):
        n, c, h, w = x.shape
        x = rearrange(x, "n c h w -> n (c h w)")
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.sigmoid(x)
        x = self.layers[-1](x)
        x = rearrange(x, "n (c h w a) -> n a c h w", n=n, c=c, h=h, w=w, a=self.n_class)
        return x


class MADE(BaseModel):
    def __init__(
        self, 
        datamodule, 
        hidden_dim, 
        n_layer,
        lr=1e-3
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()
        self.model = MADENet(self.width*self.height*self.channels, hidden_dim, n_class=256, n_layer=n_layer)
        
        self.register_buffer("log2", torch.log(torch.tensor(2, dtype=torch.float32, device=self.device)))

    def forward(self, x, y=None):
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor in range (0, 1).
            y - one-hot vector indicating class label.
        """
        logits = self.model(x)
        return logits 

    def calc_likelihood(self, x, label=None):
        # Forward pass with bpd likelihood calculation
        pred = self.forward(x, label)
        if self.input_normalize:
            target = ((x + 1) / 2 * 255).to(torch.long)
        else:
            target = (x * 255).to(torch.long)
        nll = F.cross_entropy(pred, target, reduction="none")  # (N, C, H, W)
        bpd = nll.mean(dim=[1, 2, 3]) / self.log2
        return bpd.mean()

    @torch.no_grad()
    def sample(self, img_shape, cond=None, img=None):
        """
        Sampling function for the autoregressive model.
        Inputs:
            img_shape - Shape of the image to generate (B,C,H,W)
            img (optional) - If given, this tensor will be used as
                             a starting image. The pixels to fill
                             should be -1 in the input tensor.
        """
        # Create empty image
        if img is None:
            img = torch.zeros(img_shape, dtype=torch.float32).to(self.device) - 1
        # Generation loop
        N, C, H, W = img_shape
        for h in tqdm(range(H), leave=False):
            for w in range(W):
                # Skip if not to be filled (-1)
                if (img[:, :, h, w] != -1).all().item():
                    continue
                # For efficiency, we only have to input the upper part of the image
                # as all other parts will be skipped by the masked convolutions anyways
                pred = self.forward(img, cond) # (N, classes, C, H, W)
                probs = F.softmax(rearrange(pred[:, :, :, h, w], "n a c -> n c a"), dim=-1).reshape(N*C, 256) # (NC, n_classes)
                new_pred = torch.multinomial(probs, num_samples=1).squeeze(dim=-1).to(torch.float32) / 255
                if self.input_normalize:
                    new_pred = new_pred*2 - 1
                img[:, :, h, w] = new_pred.reshape(N, C)
        return img

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
            sample_img = self.sample(img.shape)
        return ValidationResult(real_image=img, fake_image=sample_img)
