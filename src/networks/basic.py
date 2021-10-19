from torch import nn
import torch

from src.networks.conv64 import get_norm_layer
from .base import BaseNetwork
from .utils import FeatureExtractor


class LinearAct(nn.Module):
    def __init__(
        self, input_channel, output_channel, act="relu", dropout=0, batch_norm=False
    ):
        super().__init__()

        self.fc = nn.Linear(input_channel, output_channel)
        self.dropout = nn.Dropout(dropout)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == "identity":
            self.act = nn.Identity()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "tanh":
            self.act = nn.Tanh()
        else:
            raise NotImplementedError
        if batch_norm:
            self.bn = nn.BatchNorm1d(output_channel)
        else:
            self.bn = nn.Identity()

    def forward(self, x):
        # NOTE: batch_norm should placed before activation, othervise netD will not converge
        return self.dropout(self.act(self.bn(self.fc(x))))


class MLPEncoder(BaseNetwork):
    def __init__(
        self,
        input_channel,
        output_channel,
        hidden_dims,
        width,
        height,
        dropout=0,
        batch_norm=True,
        return_features=False,
        first_batch_norm=False,
        last_batch_norm=False,
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
                batch_norm=first_batch_norm,
            ),
            *[
                LinearAct(x, y, "leaky_relu", dropout=dropout, batch_norm=batch_norm)
                for x, y in zip(hidden_dims[:-1], hidden_dims[1:])
            ],
        )
        self.feature_extractor(self.model)
        self.classifier = LinearAct(
            hidden_dims[-1], output_channel, output_act, batch_norm=last_batch_norm
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
        batch_norm=True,
    ):
        super().__init__(input_channel, output_channel)
        self.width = width
        self.height = height

        dims = [input_channel, *hidden_dims]
        self.model = nn.Sequential(
            *[
                LinearAct(x, y, "relu", batch_norm=batch_norm)
                for x, y in zip(dims[:-1], dims[1:])
            ],
            LinearAct(
                hidden_dims[-1],
                output_channel * width * height,
                act=output_act,
                batch_norm=False,
            ),
        )

    def forward(self, x):
        return self.model(x).reshape(-1, self.output_channel, self.width, self.height)


class ConvDecoder(BaseNetwork):
    def __init__(self, input_channel, output_channel, ngf, batch_norm=True):
        super().__init__(input_channel, output_channel)
        self.network = nn.Sequential(
            nn.ConvTranspose2d(input_channel, ngf * 4, 4, 1, 0, bias=False),
            get_norm_layer(batch_norm)(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, output_channel, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        N = x.shape[0]
        x = x.reshape(N, -1, 1, 1)
        output = self.network(x)
        return output


class ConvEncoder(BaseNetwork):
    def __init__(
        self, input_channel, output_channel, ndf, batch_norm=True, return_features=False
    ):
        super().__init__(input_channel, output_channel)
        self.return_features = return_features
        if return_features:
            self.feature_extractor = FeatureExtractor()
        else:
            self.feature_extractor = lambda x: x
        self.output_channel = output_channel
        self.network = nn.Sequential(
            nn.Conv2d(input_channel, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ndf * 4),
            self.feature_extractor(nn.LeakyReLU(0.2, inplace=True)),
            nn.Conv2d(ndf * 4, output_channel, 4, 1, 0, bias=False),
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
