from torch import nn
from .base import BaseNetwork


def get_norm_layer(batch_norm=True):
    if batch_norm:
        return nn.BatchNorm2d
    else:
        return nn.Identity


class Decoder(BaseNetwork):
    def __init__(self, input_channel=1, output_channel=3, ngf=32, batch_norm=True):
        super().__init__(input_channel, output_channel)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(input_channel, ngf * 8, 2, 1, 0, bias=False),
            get_norm_layer(batch_norm)(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8 
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, output_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32 
        )

    def forward(self, input):
        N = input.shape[0]
        input = input.reshape(N, -1, 1, 1)
        return self.main(input)


class Encoder(BaseNetwork):
    def __init__(self, input_channel, output_channel, ndf, batch_norm=True):
        super().__init__(input_channel, output_channel)
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(input_channel, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            get_norm_layer(batch_norm)(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2 x 2
            nn.Conv2d(ndf * 8, output_channel, 2, 1, 0, bias=False),
        )

    def forward(self, input):
        N = input.shape[0]
        return self.main(input).reshape(N, -1)
