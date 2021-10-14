import torch
from torch import nn
from abc import abstractmethod


class ShapeChecker:
    def __init__(self, input_channel, output_channel) -> None:
        self.input_channel = input_channel
        self.output_channel = output_channel

    def __call__(self, module, input, output):
        assert input[0].shape[1] == self.input_channel
        assert output.shape[1] == self.output_channel


class BaseNetwork(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
