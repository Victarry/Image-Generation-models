from dataclasses import dataclass
from pytorch_lightning import LightningModule
from src.utils.utils import get_logger
import torch
import matplotlib.pyplot as plt
import io
import PIL
from torchvision.transforms import ToTensor
import numpy as np
import torch.nn.functional as F

@dataclass
class ValidationResult():
    real_image: torch.Tensor = None
    fake_image: torch.Tensor = None
    recon_image: torch.Tensor = None
    label: torch.Tensor = None
    encode_latent: torch.Tensor = None

class BaseModel(LightningModule):
    def __init__(self, datamodule) -> None:
        super().__init__()
        self.console = get_logger()
        self.width = datamodule.width
        self.height = datamodule.height
        self.channels = datamodule.channels
        self.input_normalize = datamodule.transforms.normalize
        if self.input_normalize:
            self.output_act = "tanh"
        else:
            self.output_act = "sigmoid"

    def sample(self, N: int):
        z = torch.randn(N, self.hparams.latent_dim).to(self.device)
        return self.forward(z)

    def plot_scatter(self, name, x, y, c=None, s=None, xlim=None, ylim=None):
        x, y, c, s = tensor_to_array(x, y, c, s)

        plt.figure()
        plt.scatter(x=x, y=y, s=s, c=c, cmap="tab10", alpha=1)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.title("Latent distribution")
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        plt.close()
        buf.seek(0)
        visual_image = ToTensor()(PIL.Image.open(buf))
        self.logger.experiment.add_image(name, visual_image, self.global_step)

def tensor_to_array(*tensors):
    output = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            output.append(np.array(tensor.detach().cpu().numpy()))
        else:
            output.append(tensor)
    return output