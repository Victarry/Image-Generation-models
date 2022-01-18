from typing import ForwardRef
from torch import nn
import torch
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .base import BaseModel
from pathlib import Path
import torchvision


class MaskedConvolution(nn.Module):
    def __init__(self, c_in, c_out, mask, **kwargs):
        super().__init__()
        self.register_buffer("mask", mask)
        kernel_size = mask.shape
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple([dilation * (kernel_size[i] - 1) // 2 for i in range(2)])
        # Actual convolution
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, **kwargs)

    def forward(self, x):
        self.conv.weight.data *= self.mask
        return self.conv(x)


class VerticalStackConvolution(MaskedConvolution):
    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size // 2 + 1 :, :] = 0
        if mask_center:
            mask[kernel_size // 2] = 0
        super().__init__(c_in, c_out, mask, **kwargs)


class HorizontalStackConvolution(MaskedConvolution):
    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        mask = torch.ones(1, kernel_size)
        mask[0, kernel_size // 2 + 1 :] = 0
        if mask_center:
            mask[0, kernel_size // 2] = 0
        super().__init__(c_in, c_out, mask, **kwargs)


class GatedMaskedConv(nn.Module):
    def __init__(self, channels, kernel_size=3, **kwargs):
        super().__init__()
        self.horiz_conv = HorizontalStackConvolution(
            channels, 2 * channels, kernel_size, mask_center=False
        )
        self.vert_conv = VerticalStackConvolution(
            channels, 2 * channels, kernel_size, mask_center=False
        )
        self.conv1x1_1 = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.conv1x1_2 = nn.Conv2d(channels, channels, 1)
        self.output_channels = channels
        self.input_channels = channels

    def forward(self, vert_x, horiz_x):
        vert_conv_x = self.vert_conv(vert_x)
        vert_x1, vert_x2 = torch.chunk(vert_conv_x, 2, dim=1)
        out_vert_x = torch.tanh(vert_x1) * torch.sigmoid(vert_x2)

        horiz_x1, horiz_x2 = torch.chunk(
            self.horiz_conv(horiz_x) + self.conv1x1_1(vert_conv_x), 2, 1
        )
        out_horiz_x = torch.tanh(horiz_x1) * torch.tanh(horiz_x2)
        out_horiz_x = self.conv1x1_2(out_horiz_x) + horiz_x

        return out_vert_x, out_horiz_x


class PixelCNN(BaseModel):
    def __init__(
        self, channels, width, height, hidden_dim, input_normalize=True, lr=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(
            channels, hidden_dim, 5, mask_center=True
        )
        self.conv_hstack = HorizontalStackConvolution(
            channels, hidden_dim, 5, mask_center=True
        )
        # Convolution block of PixelCNN. We use dilation instead of downscaling
        self.conv_layers = nn.ModuleList(
            [
                GatedMaskedConv(hidden_dim),
                GatedMaskedConv(hidden_dim, dilation=2),
                GatedMaskedConv(hidden_dim),
                GatedMaskedConv(hidden_dim, dilation=4),
                GatedMaskedConv(hidden_dim),
                GatedMaskedConv(hidden_dim, dilation=2),
                GatedMaskedConv(hidden_dim),
                # GatedMaskedConv(hidden_dim, dilation=4),
                # GatedMaskedConv(hidden_dim),
                # GatedMaskedConv(hidden_dim, dilation=2),
                # GatedMaskedConv(hidden_dim),
            ]
        )
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv2d(hidden_dim, channels * 256, kernel_size=1, padding=0)

    def forward(self, x):
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor in range (0, 1).
        """
        # Scale input from 0 to 255 back to -1 to 1
        if self.hparams.input_normalize:
            x = x * 2 - 1
        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(F.elu(h_stack))

        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        out = out.reshape(
            out.shape[0], 256, out.shape[1] // 256, out.shape[2], out.shape[3]
        )
        return out

    def calc_likelihood(self, x):
        # Forward pass with bpd likelihood calculation
        pred = self.forward(x)
        target = (x * 255).to(torch.long)
        nll = F.cross_entropy(pred, target, reduction="none")  # (N, C, H, W)
        bpd = nll.mean(dim=[1, 2, 3]) * np.log2(np.exp(1))
        return bpd.mean()

    @torch.no_grad()
    def sample(self, img_shape, img=None):
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
        for h in tqdm(range(img_shape[2]), leave=False):
            for w in range(img_shape[3]):
                for c in range(img_shape[1]):
                    # Skip if not to be filled (-1)
                    if (img[:, c, h, w] != -1).all().item():
                        continue
                    # For efficiency, we only have to input the upper part of the image
                    # as all other parts will be skipped by the masked convolutions anyways
                    pred = self.forward(img[:, :, : h + 1, :])
                    probs = F.softmax(pred[:, :, c, h, w], dim=-1)
                    img[:, c, h, w] = (
                        torch.multinomial(probs, num_samples=1)
                        .squeeze(dim=-1)
                        .to(torch.float32)
                        / 255
                    )
        if self.hparams.input_normalize:
            return img * 2 - 1
        else:
            return img

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss = self.calc_likelihood(batch[0])
        self.log("train_bpd", loss)
        return loss

    def on_train_epoch_end(self):
        result_path = Path("results")
        result_path.mkdir(parents=True, exist_ok=True)
        imgs = self.sample(
            (64, self.hparams.channels, self.hparams.width, self.hparams.height)
        )
        grid = self.get_grid_images(imgs)
        torchvision.utils.save_image(grid, result_path / f"{self.current_epoch}.jpg")

    def validation_step(self, batch, batch_idx):
        loss = self.calc_likelihood(batch[0])
        self.log("val_bpd", loss)

    def test_step(self, batch, batch_idx):
        loss = self.calc_likelihood(batch[0])
        self.log("test_bpd", loss)
