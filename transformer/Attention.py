import torch.nn as nn

# TODO: Masked
# TODO: To parallelize attention with multi-heads
class Attention(nn.Module):
    def __init__(self, input_dim, att_dim, heads=1, masked=False):
        super().__init__()
        self.attention = []
        # To parallelize attention with multi-heads
        for head in range(heads):
            Q = nn.Linear(input_dim, att_dim, bias=True)
            K = nn.Linear(input_dim, att_dim, bias=True)
            V = nn.Linear(input_dim, att_dim, bias=True)
            self.attention.append([Q, K, V])
        
    def forward(self, q, k, v):
        pass