from torch import nn
import torch
from torch.nn.modules.activation import LeakyReLU

class LinearAct(nn.Module):
    def __init__(self, input_dim, output_dim, act='relu'):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            self.act = nn.LeakyReLU(inplace=True)
        elif act == 'identity':
            self.act = nn.Identity()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise NotImplementedError
    
    def forward(self, x):
        return self.act(self.fc(x))


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()

        dims = [input_dim, *hidden_dims]
        self.model = nn.Sequential(
            *[LinearAct(x, y, 'relu') for x,y in zip(dims[:-1], dims[1:])],
            LinearAct(hidden_dims[-1], output_dim, 'identity')
        )
    
    def forward(self, x):
        N = x.shape[0]
        x = x.reshape(N, -1)
        return self.model(x)


class MLPDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()

        dims = [input_dim, *hidden_dims]
        self.model = nn.Sequential(
            *[LinearAct(x, y, 'leaky_relu') for x,y in zip(dims[:-1], dims[1:])],
            LinearAct(hidden_dims[-1], output_dim, 'tanh')
        )
    
    def forward(self, x):
        return self.model(x)
