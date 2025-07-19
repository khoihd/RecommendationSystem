import torch.nn as nn

# TODO: Masked
# TODO: Multihead
class Attention(nn.Module):
    def __init__(self, heads=1, masked=True):
        super().__init__()
        self.attention = []
        for head in range(heads):
            Q = nn.Linear()
            K = nn.Linear()
            V = nn.Linear()
            self.attention.append([Q, K, V])
        
    def forward(self, q, k, v):
        pass