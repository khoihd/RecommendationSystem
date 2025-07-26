import torch.nn as nn


class AddNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x, y):
        return self.layernorm(x + y)
    