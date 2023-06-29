import torch
import torch.nn as nn


class AEModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(AEModel, self).__init__()
        self.architecture = architecture
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        current_dim = input_dim
        for i, layer in enumerate(self.architecture):
            next_dim = layer
            if i == len(self.architecture) - 1:
                self.encoder.append(nn.Sequential(nn.Linear(current_dim, next_dim)))
            else:
                self.encoder.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
                )
                current_dim = next_dim

        last_dim = input_dim
        current_dim = self.architecture[-1]
        for i, layer in enumerate(reversed(self.architecture[:-1])):
            next_dim = layer
            self.decoder.append(
                nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
            )
            current_dim = next_dim
        self.decoder.append(nn.Sequential(nn.Linear(current_dim, last_dim)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x
