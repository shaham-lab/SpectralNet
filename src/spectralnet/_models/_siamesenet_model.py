import torch
import torch.nn as nn


class SiameseNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(SiameseNetModel, self).__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList()

        current_dim = input_dim
        for layer in self.architecture:
            next_dim = layer
            self.layers.append(
                nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
            )
            current_dim = next_dim

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2
