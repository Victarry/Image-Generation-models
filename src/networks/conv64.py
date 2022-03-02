from torch import nn
import torch
from .base import BaseNetwork
from .utils import FeatureExtractor
from .basic import get_act_function, get_norm_layer



class Decoder(BaseNetwork):
    def __init__(self, input_channel=1, output_channel=3, ngf=32, batch_norm=True, output_act="tanh"):
        super().__init__(input_channel, output_channel)
        self.main = nn.Sequential(
            # (input_channel) x 1 x 1
            nn.Conv2d(input_channel, ngf*8, 1, 1, 0, bias=False),
            get_norm_layer(batch_norm)(ngf * 8),
            nn.ReLU(True),
            # (ngf*8) x 1 x 1
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 1, 0, bias=False),
            get_norm_layer(batch_norm)(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, output_channel, 4, 2, 1, bias=False),
            get_act_function(output_act)
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        N = input.shape[0]
        input = input.reshape(N, -1, 1, 1)
        return self.main(input)


class Encoder(BaseNetwork):
    def __init__(
        self, input_channel, output_channel, ndf, batch_norm=True, return_features=False
    ):
        super().__init__(input_channel, output_channel)
        self.return_features = return_features

        if return_features:
            self.feature_extractor = FeatureExtractor()
        else:
            self.feature_extractor = lambda x: x
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(input_channel, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ndf * 4),
            self.feature_extractor(nn.LeakyReLU(0.2, inplace=True)),  # extract features
            # state size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 0, bias=False),
            get_norm_layer(batch_norm)(ndf * 8),
            # state size: (ndf*8) x 1 x 1
            nn.Conv2d(ndf * 8, output_channel, 1, 1, 0, bias=True),
        )

    def forward(self, input):
        N = input.shape[0]
        if self.return_features:
            self.feature_extractor.clean()
            output = self.main(input).reshape(N, -1)
            features = torch.cat(
                [torch.ravel(x) for x in self.feature_extractor.features]
            )
            return output, features
        else:
            output = self.main(input).reshape(N, -1)
            return output
