from functools import partial
from types import new_class
from torch import nn
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from src.models.base import BaseModel, ValidationResult

# TODO: 
# 1. handle color dependency problem(which is nontrival with mask handling), thus has worse results on color images.
# 2. Parallen PixelCNN
# 3. 

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
    def __init__(self, channels, kernel_size=3, cond_channel=None, **kargs):
        super().__init__()
        self.horiz_conv = HorizontalStackConvolution(
            channels, 2 * channels, kernel_size, mask_center=False, **kargs
        )
        self.vert_conv = VerticalStackConvolution(
            channels, 2 * channels, kernel_size, mask_center=False, **kargs
        )
        self.conv1x1_1 = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.conv1x1_2 = nn.Conv2d(channels, channels, 1)
        self.output_channels = channels
        self.input_channels = channels
        if cond_channel is not None:
            self.cond_proj_vert1 = nn.Conv2d(cond_channel, channels, kernel_size=1, bias=False)
            self.cond_proj_vert2 = nn.Conv2d(cond_channel, channels, kernel_size=1, bias=False)
            self.cond_proj_horiz1 = nn.Conv2d(cond_channel, channels, kernel_size=1, bias=False)
            self.cond_proj_horiz2 = nn.Conv2d(cond_channel, channels, kernel_size=1, bias=False)

    def forward(self, vert_x, horiz_x, cond=None):
        # The horizontal branch can take information from vertical branch, while not vice versa. Think it carefully.
        vert_conv_x = self.vert_conv(vert_x)
        vert_x1, vert_x2 = torch.chunk(vert_conv_x, 2, dim=1)
        if cond is None:
            out_vert_x = torch.tanh(vert_x1) * torch.sigmoid(vert_x2)
        else:
            out_vert_x = torch.tanh(vert_x1 + self.cond_proj_vert1(cond).expand_as(vert_x1)) * torch.sigmoid(vert_x2 + self.cond_proj_vert2(cond).expand_as(vert_x2))

        horiz_x1, horiz_x2 = torch.chunk(
            self.horiz_conv(horiz_x) + self.conv1x1_1(vert_conv_x), 2, 1
        )
        if cond is None:
            out_horiz_x = torch.tanh(horiz_x1) * torch.tanh(horiz_x2)
        else:
            out_horiz_x = torch.tanh(horiz_x1 + self.cond_proj_horiz1(cond).expand_as(horiz_x1)) * torch.tanh(horiz_x2 + self.cond_proj_horiz2(cond).expand_as(horiz_x2))
        out_horiz_x = self.conv1x1_2(out_horiz_x) + horiz_x

        return out_vert_x, out_horiz_x


class PixelCNN(BaseModel):
    def __init__(
        self, 
        datamodule, 
        hidden_dim, 
        class_condition=False,
        n_classes=None,
        lr=1e-3
    ):
        super().__init__(datamodule)
        self.save_hyperparameters()

        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(
            self.channels, hidden_dim, 5, mask_center=True
        )
        self.conv_hstack = HorizontalStackConvolution(
            self.channels, hidden_dim, 5, mask_center=True
        )
        # Convolution block of PixelCNN. We use dilation instead of downscaling
        if class_condition:
            conv_layer = partial(GatedMaskedConv, cond_channel=n_classes)
        else:
            conv_layer = GatedMaskedConv
        self.conv_layers = nn.ModuleList(
            [
                conv_layer(hidden_dim),
                conv_layer(hidden_dim, dilation=2),
                conv_layer(hidden_dim),
                conv_layer(hidden_dim, dilation=4),
                conv_layer(hidden_dim),
                conv_layer(hidden_dim, dilation=2),
                conv_layer(hidden_dim),
                conv_layer(hidden_dim, dilation=4),
                conv_layer(hidden_dim),
                conv_layer(hidden_dim, dilation=2),
                conv_layer(hidden_dim),
            ]
        )
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv2d(hidden_dim, self.channels * 256, kernel_size=1, padding=0)
        self.register_buffer("log2", torch.log(torch.tensor(2, dtype=torch.float32, device=self.device)))

    def forward(self, x, y=None):
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor in range (0, 1).
            y - one-hot vector indicating class label.
        """
        N = x.shape[0]
        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            if y is not None:
                y = y.reshape(N, self.hparams.n_classes, 1, 1)
                v_stack, h_stack = layer(v_stack, h_stack, y)
            else:
                v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(F.elu(h_stack))

        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        out = out.reshape(
            out.shape[0], 256, out.shape[1] // 256, out.shape[2], out.shape[3]
        )
        return out

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
                pred = self.forward(img[:, :, : h + 1, :], cond) # (N, classes, C)
                probs = F.softmax(pred[:, :, :, h, w].permute(0, 2, 1), dim=-1).reshape(N*C, -1)
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
        if self.hparams.class_condition:
            label = F.one_hot(label, num_classes=self.hparams.n_classes).to(torch.float32)
            loss = self.calc_likelihood(img, label)
        else:
            loss = self.calc_likelihood(img)
        self.log("train_bpd", loss) # bpd: bits per dim, by entropy encoding
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        N, C, H, W = img.shape
        if self.hparams.class_condition:
            label = F.one_hot(label, num_classes=self.hparams.n_classes).to(torch.float32)
            loss = self.calc_likelihood(img, label)
        else:
            loss = self.calc_likelihood(img)
        self.log("val_bpd", loss)

        sample_img = None
        if batch_idx == 0:
            if self.hparams.class_condition:
                sample_label = torch.arange(self.hparams.n_classes, device=self.device).reshape(self.hparams.n_classes, 1).repeat(1, 8)
                sample_label = F.one_hot(sample_label, num_classes=self.hparams.n_classes).to(torch.float32)
                sample_img = self.sample((self.hparams.n_classes*8, C, H, W), cond=sample_label)
            else:
                sample_img = self.sample(img.shape)
        return ValidationResult(real_image=img, fake_image=sample_img)
