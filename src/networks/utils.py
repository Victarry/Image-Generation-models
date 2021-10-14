from torch import nn


class FeatureExtractor:
    def __init__(self) -> None:
        super().__init__()
        self.features = []

    def __call__(self, module: nn.Module):
        module.register_forward_hook(self.forward_hook())
        return module

    def forward_hook(self):
        def fn(module, input, output):
            self.features.append(output)

        return fn

    def clean(self):
        self.features = []
