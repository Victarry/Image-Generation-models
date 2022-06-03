from dataclasses import dataclass, field
from pytorch_lightning import LightningModule
from src.utils.utils import get_logger
import torch
import torch.nn.functional as F

@dataclass
class ValidationResult():
    others: dict = field(default_factory=dict)
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

