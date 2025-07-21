import math
import torch
import torch.nn as nn

class PositionalWordEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.word_embeddding = nn.Embedding(vocab_size, emb_dim)
    
    def forward(self, x):
        return self.word_embeddding(x) + self.positional_embeddding(x)

    # x: batch x seq_len
    def positional_embeddding(self, x):
        batch = x.size()[0]
        
        # dimensions: emb_dim
        dimensions = torch.arange(0, self.emb_dim, 2)
        # frequency: emb_dim
        frequency = torch.exp(dimensions * (-math.log(10000)) / self.emb_dim)
        
        # pos_emb: max_len
        pos_emb = torch.arange(0, self.max_len).unsqueeze(1)
        
        # pe: max_len x emb_dim
        pe = torch.zeros(self.max_len, self.emb_dim)
        pe[:,0::2] = torch.sin(pos_emb * frequency)
        pe[:,1::2] = torch.cos(pos_emb * frequency)

        # pe: batch_size x 
        return pe.repeat(batch, 1, 1)