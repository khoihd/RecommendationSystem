import math
import torch
import torch.nn as nn


class PositionalWordEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.word_embeddding = nn.Embedding(vocab_size, emb_dim)
        self.register_buffer('pe', torch.zeros(emb_dim))    # seq_len is only known in forward

    def forward(self, x):
        return self.word_embeddding(x) + self.positional_embeddding(x)

    # x: batch x seq_len
    def positional_embeddding(self, x):
        batch = x.size()[0]
        seq_len = x.size()[1]

        # dimensions: emb_dim
        dimensions = torch.arange(0, self.emb_dim, 2)
        # frequency: emb_dim
        frequency = torch.exp(dimensions * (-math.log(10000)) / self.emb_dim)

        # pos_emb: seq_len
        pos_emb = torch.arange(0, seq_len).unsqueeze(1)

        # pe: seq_len x emb_dim
        self.pe = self.pe.repeat(seq_len, 1)
        self.pe[:, 0::2] = torch.sin(pos_emb * frequency)
        self.pe[:, 1::2] = torch.cos(pos_emb * frequency)

        return self.pe.repeat(batch, 1, 1)
