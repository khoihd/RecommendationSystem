import numpy as np
import torch
import torch.nn as nn

# TODO: How to test Attention
class Attention(nn.Module):
    def __init__(self, input_dim, attn_heads):
        super().__init__()
        assert input_dim % attn_heads == 0, "input_dim must be divided by attn_heads"
        attn_dim = input_dim // attn_heads
        
        self.attn_dim = attn_dim
        self.attn_heads = attn_heads

        self.Q = nn.Linear(input_dim, attn_heads * attn_dim, bias=True)
        self.K = nn.Linear(input_dim, attn_heads * attn_dim, bias=True)
        self.V = nn.Linear(input_dim, attn_heads * attn_dim, bias=True)    
    
    # queries:  batch x query_len x input_dim
    # keys:     batch x key_len x input_dim
    # values:   batch x value_len x input_dim
    def forward(self, q, k, v, mask=False):
        batch_size = q.size(0)
        query_len = q.size(1)
        keyvalue_len = k.size(1)

        attn_q = self.Q(q)
        attn_k = self.K(k)
        attn_v = self.V(v)

        # attn_q: batch x attn_heads x query_len x attn_dim
        # attn_k: batch x attn_heads x keyvalue_len x attn_dim
        # attn_v: batch x attn_heads x keyvalue_len x attn_dim
        attn_q = attn_q.reshape(batch_size, query_len, self.attn_heads, self.attn_dim).transpose(1, 2)
        attn_k = attn_k.reshape(batch_size, keyvalue_len, self.attn_heads, self.attn_dim).transpose(1, 2)
        attn_v = attn_v.reshape(batch_size, keyvalue_len, self.attn_heads, self.attn_dim).transpose(1, 2)

        # (batch x attn_heads x query_len x attn_dim)
        #   @ (batch x attn_heads x attn_dim x keyvalue_len)
        #  = batch x attn_heads x query_len x keyvalue_len
        attn_scores = attn_q @ attn_k.transpose(-2, -1)
        attn_scores = attn_scores / self.attn_dim**0.5

        if mask:
            mask_attn_matrix = torch.triu(torch.ones_like(attn_scores, dtype=torch.bool), 
                                            diagonal=1,
                                        )
            attn_scores = attn_scores.masked_fill(mask_attn_matrix, float('-inf'))

        # batch x attn_heads x query_len x keyvalue_len
        attn_weights = torch.softmax(attn_scores, -1)
        # (batch x attn_heads x query_len x keyvalue_len) @ (batch x attn_heads x keyvalue_len x attn_dim)
        # batch x attn_heads x query_len x attn_dim
        attn = attn_weights @ attn_v
        # batch x attn_heads x query_len x attn_dim
        attn = attn.transpose(1, 2).contiguous()
        # batch x query_len x attn_heads x attn_dim
        attn = attn.view(batch_size, query_len, self.attn_heads * self.attn_dim)
        return attn
