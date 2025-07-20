import torch
import torch.nn as nn

# TODO: Masked
# TODO: To parallelize attention with multi-heads
class Attention(nn.Module):
    def __init__(self, input_dim, attn_dim, head_count=1, masked=False):
        super().__init__()
        self.attn_heads = []
        # To parallelize attention with multi-heads
        for _ in range(head_count):
            Q = nn.Linear(input_dim, attn_dim, bias=True)
            K = nn.Linear(input_dim, attn_dim, bias=True)
            V = nn.Linear(input_dim, attn_dim, bias=True)
            self.attn_heads.append([Q, K, V])

    # queries:  batch x query_len x input_dim
    # keys:     batch x key_len x input_dim
    # values:   batch x value_len x input_dim
    def forward(self, q, k, v):
        # Each head:    batch x query_len x attn_dim
        # Concat head:  batch x query_len x (attn_dim x heads)
        result = []
        for head in self.attn_heads:
            Q, K, V = head
            attn_q = Q(q) # batch x query_len x attn_dim
            attn_k = K(k) # batch x keyvalue_len x attn_dim
            attn_v = V(v) # batch x keyvalue_len x attn_dim

            # (batch x query_len x attn_dim) @ (batch x attn_dim x keyvalue_len)
            # batch x query_len x keyvalue_len
            attn_scores = attn_q @ attn_k.permute(0, 2, 1)
            # batch x query_len x keyvalue_len
            attn_weights = torch.softmax(attn_scores, -1)
            # (batch x query_len x keyvalue_len) @ (batch x keyvalue_len x attn_dim)
            # batch x query_len x attn_dim
            attn = attn_weights @ attn_v
            result.append(attn)
        
        return torch.concat(result, dim=-1)