import math
import torch
import torch.nn as nn


class PositionalWordEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_seq_len):
        super().__init__()
        self.emb_dim = emb_dim
        self.word_embeddding = nn.Embedding(vocab_size, emb_dim)
        # max_seq_len x emb_dim
        self.register_buffer('pe', torch.zeros(max_seq_len, emb_dim))

        # dimensions: emb_dim
        dimensions = torch.arange(0, emb_dim, 2)
        # frequency: emb_dim
        frequency = torch.exp(dimensions * (-math.log(10000)) / emb_dim)

        # pos_emb: max_seq_len
        pos_emb = torch.arange(0, max_seq_len).unsqueeze(1)

        # pe: max_seq_len x emb_dim
        # reset using self.pe
        self.pe[:, 0::2] = torch.sin(pos_emb * frequency)
        self.pe[:, 1::2] = torch.cos(pos_emb * frequency)

    def forward(self, x):
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        
        return self.word_embeddding(x) + self.pe[:seq_len].repeat(batch_size, 1, 1)