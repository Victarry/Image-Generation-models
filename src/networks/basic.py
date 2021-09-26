from torch import nn


class LinearAct(nn.Module):
    def __init__(self, input_dim, output_dim, act="relu", dropout=0, batch_norm=False):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
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
            self.bn = nn.BatchNorm1d(output_dim)
        else:
            self.bn = nn.Identity()

    def forward(self, x):
        # NOTE: batch_norm should placed before activation, othervise netD will not converge
        return self.dropout(self.act(self.bn(self.fc(x))))


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0, batch_norm=False):
        super().__init__()

        dims = [input_dim, *hidden_dims]
        self.model = nn.Sequential(
            # first layer not use batch_norm
            LinearAct(
                input_dim,
                hidden_dims[0],
                "leaky_relu",
                dropout=dropout,
                batch_norm=False,
            ),
            *[
                LinearAct(x, y, "leaky_relu", dropout=dropout, batch_norm=batch_norm)
                for x, y in zip(dims[1:-1], dims[2:])
            ],
            LinearAct(hidden_dims[-1], output_dim, "identity", batch_norm=False)
        )

    def forward(self, x):
        N = x.shape[0]
        x = x.reshape(N, -1)
        return self.model(x)


class MLPDecoder(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dims, output_act, batch_norm=False
    ):
        super().__init__()

        dims = [input_dim, *hidden_dims]
        self.model = nn.Sequential(
            *[
                LinearAct(x, y, "relu", batch_norm=batch_norm)
                for x, y in zip(dims[:-1], dims[1:])
            ],
            LinearAct(hidden_dims[-1], output_dim, act=output_act, batch_norm=False)
        )

    def forward(self, x):
        return self.model(x)


class ConvGenerator(nn.Module):
    def __init__(self, latent_dim, output_channel, ngf):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, output_channel, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        N = input.shape[0]
        output = self.network(input.reshape(N, -1, 1, 1))
        return output


class ConvDiscriminator(nn.Module):
    def __init__(self, input_channel, output_channel, ndf):
        super().__init__()
        self.output_channel = output_channel
        self.network = nn.Sequential(
            nn.Conv2d(input_channel, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, output_channel, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        output = self.network(input)
        return output.view(-1, self.output_channel)
