# %%
import matplotlib
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.models.pixelcnn import HorizontalStackConvolution, VerticalStackConvolution

# %% Show receptive field of convolution
def show_center_recep_field(img, out):
    """
    Calculates the gradients of the input with respect to the output center pixel,
    and visualizes the overall receptive field.
    Inputs:
        img - Input image for which we want to calculate the receptive field on.
        out - Output features/loss which is used for backpropagation, and should be
              the output of the network/computation graph.
    """
    # Determine gradients
    loss = out[0, :, img.shape[2] // 2, img.shape[3] // 2].sum()  # L1 loss
    # Retain graph as we want to stack multiple layers and show the receptive field of all of them
    loss.backward(retain_graph=True)

    img_grads = img.grad.abs()
    img.grad.fill_(0)  # Reset grads

    # Plot receptive field
    img = img_grads.squeeze().cpu().numpy()  # (H, W)
    fig, ax = plt.subplots(1, 2)
    pos = ax[0].imshow(img)  # weighted receptive field
    ax[1].imshow(img > 0)  # binary receptive field
    # Mark the center pixel in red if it doesn't have any gradients (should be the case for standard autoregressive models)
    show_center = img[img.shape[0] // 2, img.shape[1] // 2] == 0
    if show_center:
        center_pixel = np.zeros(img.shape + (4,))
        center_pixel[
            center_pixel.shape[0] // 2, center_pixel.shape[1] // 2, :
        ] = np.array([1.0, 0.0, 0.0, 1.0])
    for i in range(2):
        ax[i].axis("off")
        if show_center:
            ax[i].imshow(center_pixel)
    ax[0].set_title("Weighted receptive field")
    ax[1].set_title("Binary receptive field")
    plt.show()
    plt.close()


# %%
inp_img = torch.zeros(1, 1, 11, 11)
inp_img.requires_grad_()
show_center_recep_field(inp_img, inp_img)

# %% show horizontal convolution ERF
horiz_conv = HorizontalStackConvolution(
    c_in=1, c_out=1, kernel_size=3, mask_center=True
)
horiz_conv.conv.weight.data.fill_(1)
horiz_conv.conv.bias.data.fill_(0)
horiz_img = horiz_conv(inp_img)
show_center_recep_field(inp_img, horiz_img)

# %% show vertical convolution ERF
vert_conv = VerticalStackConvolution(c_in=1, c_out=1, kernel_size=3, mask_center=True)
vert_conv.conv.weight.data.fill_(1)
vert_conv.conv.bias.data.fill_(0)
vert_img = vert_conv(inp_img)
show_center_recep_field(inp_img, vert_img)

# %%
# Initialize convolutions with equal weight to all input pixels
horiz_conv = HorizontalStackConvolution(
    c_in=1, c_out=1, kernel_size=3, mask_center=False
)
horiz_conv.conv.weight.data.fill_(1)
horiz_conv.conv.bias.data.fill_(0)
vert_conv = VerticalStackConvolution(c_in=1, c_out=1, kernel_size=3, mask_center=False)
vert_conv.conv.weight.data.fill_(1)
vert_conv.conv.bias.data.fill_(0)

# We reuse our convolutions for the 4 layers here. Note that in a standard network,
# we don't do that, and instead learn 4 separate convolution. As this cell is only for
# visualization purposes, we reuse the convolutions for all layers.
for l_idx in range(4):
    vert_img = vert_conv(vert_img)
    horiz_img = horiz_conv(horiz_img) + vert_img
    print(f"Layer {l_idx+2}")
    show_center_recep_field(inp_img, horiz_img)

# %%
