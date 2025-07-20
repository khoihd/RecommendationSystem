import numpy as np
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, input_dim, attn_dim, attn_head, masked):
        super().__init__()
        self.attn_dim = attn_dim
        self.attn_head = attn_head
        self.masked = masked
    
        self.Q = nn.Linear(input_dim, attn_head * attn_dim, bias=True)
        self.K = nn.Linear(input_dim, attn_head * attn_dim, bias=True)
        self.V = nn.Linear(input_dim, attn_head * attn_dim, bias=True)

    # queries:  batch x query_len x input_dim
    # keys:     batch x key_len x input_dim
    # values:   batch x value_len x input_dim
    def forward(self, q, k, v):
        batch_size = q.size()[0]
        query_len = q.size()[1]
        keyvalue_len = k.size()[1]
        
        attn_q = self.Q(q).view(batch_size, query_len, self.attn_head, self.attn_dim).transpose(1, 2)
        attn_k = self.K(k).view(batch_size, keyvalue_len, self.attn_head, self.attn_dim).transpose(1, 2)
        attn_v = self.V(v).view(batch_size, keyvalue_len, self.attn_head, self.attn_dim).transpose(1, 2)
        # attn_q: batch x attn_head x query_len x attn_dim
        # attn_k: batch x attn_head x keyvalue_len x attn_dim
        # attn_v: batch x attn_head x keyvalue_len x attn_dim

        # (batch x attn_head x query_len x attn_dim) @ (batch x attn_head x attn_dim x keyvalue_len)
        # batch x attn_head x query_len x keyvalue_len
        attn_scores = attn_q @ attn_k.transpose(-2, -1)
        attn_scores = attn_scores / self.attn_dim**0.5
        # batch x attn_head x query_len x keyvalue_len
        attn_weights = torch.softmax(attn_scores, -1)
        # (batch x attn_head x query_len x keyvalue_len) @ (batch x attn_head x keyvalue_len x attn_dim)
        # batch x attn_head x query_len x attn_dim
        attn = attn_weights @ attn_v
        # batch x attn_head x query_len x attn_dim
        attn = attn.transpose(1, 2).contiguous()
        # batch x query_len x attn_head x attn_dim
        attn = attn.view(batch_size, query_len, self.attn_head * self.attn_dim)
        return attn