from torch import nn
import torch

from .base import BaseNetwork
from .utils import FeatureExtractor

def get_act_function(act="relu"):
    if act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif act == "identity":
        return nn.Identity()
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    else:
        raise NotImplementedError

def get_norm_layer(norm_type="batch"):
    if norm_type == "batch":
        return nn.BatchNorm2d
    elif norm_type == "instance":
        return nn.InstanceNorm2d
    elif norm_type == None:
        return nn.Identity
    else:
        raise NotImplementedError(f"Norm type of {norm_type} is not implemented")

def get_norm_layer_1d(norm_type="batch"):
    if norm_type == "batch":
        return nn.BatchNorm1d
    elif norm_type == "instance":
        return nn.InstanceNorm1d
    elif norm_type == None:
        return nn.Identity
    else:
        raise NotImplementedError(f"Norm type of {norm_type} is not implemented")

class LinearAct(nn.Module):
    def __init__(
        self, input_channel, output_channel, act="relu", dropout=0, norm_type="batch"
    ):
        super().__init__()
        self.act = get_act_function(act)
        self.fc = nn.Linear(input_channel, output_channel)
        self.dropout = nn.Dropout(dropout)
        self.norm = get_norm_layer_1d(norm_type)(output_channel)

    def forward(self, x):
        # NOTE: batch_norm should be placed before activation, otherwise netD will not converge
        return self.dropout(self.act(self.norm(self.fc(x))))


class MLPEncoder(BaseNetwork):
    def __init__(
        self,
        input_channel,
        output_channel,
        hidden_dims,
        width,
        height,
        dropout=0,
        norm_type="batch",
        return_features=False,
        output_act="identity",
    ):
        super().__init__(input_channel, output_channel)
        self.return_features = return_features
        if return_features:
            self.feature_extractor = FeatureExtractor()
        else:
            self.feature_extractor = lambda x: x
        self.model = nn.Sequential(
            # first layer not use batch_norm
            LinearAct(
                input_channel * width * height,
                hidden_dims[0],
                "leaky_relu",
                dropout=dropout,
                norm_type=None,
            ),
            *[
                LinearAct(x, y, "leaky_relu", dropout=dropout, norm_type=norm_type)
                for x, y in zip(hidden_dims[:-1], hidden_dims[1:])
            ],
        )
        self.feature_extractor(self.model)
        self.classifier = LinearAct(
            hidden_dims[-1], output_channel, output_act, norm_type=None
        )

    def forward(self, x):
        N = x.shape[0]
        x = x.reshape(N, -1)
        if self.return_features:
            self.feature_extractor.clean()
            output = self.classifier(self.model(x))
            return output, torch.cat(
                [torch.ravel(x) for x in self.feature_extractor.features]
            )
        else:
            return self.classifier(self.model(x))


class MLPDecoder(BaseNetwork):
    def __init__(
        self,
        input_channel,
        output_channel,
        hidden_dims,
        width,
        height,
        output_act,
        norm_type="batch",
    ):
        super().__init__(input_channel, output_channel)
        self.width = width
        self.height = height

        dims = [input_channel, *hidden_dims]
        self.model = nn.Sequential(
            *[
                LinearAct(x, y, "relu", norm_type=norm_type)
                for x, y in zip(dims[:-1], dims[1:])
            ],
            LinearAct(
                hidden_dims[-1],
                output_channel * width * height,
                act=output_act,
                norm_type=False,
            ),
        )

    def forward(self, x):
        return self.model(x).reshape(-1, self.output_channel, self.width, self.height)


class ConvDecoder(BaseNetwork):
    def __init__(self, input_channel, output_channel, ngf, norm_type="batch", output_act="tanh"):
        super().__init__(input_channel, output_channel)
        # cause checkboard artifacts
        self.network = nn.Sequential(
            nn.ConvTranspose2d(input_channel, ngf * 4, 4, 1, 0),
            get_norm_layer(norm_type)(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1),
            get_norm_layer(norm_type)(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            get_norm_layer(norm_type)(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, output_channel, 4, 2, 1),
            get_act_function(output_act)
        )

    def forward(self, x):
        N = x.shape[0]
        x = x.reshape(N, -1, 1, 1)
        output = self.network(x)
        return output


class ConvEncoder(BaseNetwork):
    def __init__(
        self, input_channel, output_channel, ndf, norm_type="batch", return_features=False
    ):
        super().__init__(input_channel, output_channel)
        self.return_features = return_features
        if return_features:
            self.feature_extractor = FeatureExtractor()
        else:
            self.feature_extractor = lambda x: x
        self.output_channel = output_channel
        self.network = nn.Sequential(
            nn.Conv2d(input_channel, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            get_norm_layer(norm_type)(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1),
            get_norm_layer(norm_type)(ndf * 4),
            self.feature_extractor(nn.LeakyReLU(0.2, inplace=True)),
            nn.Conv2d(ndf * 4, output_channel, 4, 1, 0),
        )

    def forward(self, x):
        if self.return_features:
            self.feature_extractor.clean()
            output = self.network(x).reshape(-1, self.output_channel)
            return output, torch.cat(
                [torch.ravel(x) for x in self.feature_extractor.features]
            )
        else:
            return self.network(x).reshape(-1, self.output_channel)
