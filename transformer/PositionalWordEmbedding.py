import math
import torch
import torch.nn as nn

# TODO: How to parallelize PositionalWordEmbedding
class PositionalWordEmbedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.word_embeddding = nn.Embedding(vocab_size, dim)
    
    def forward(self, x):
        return self.word_embeddding(x) + self.positional_embeddding(x)

    # x: batch x seq_len
    def positional_embeddding(self, x):
        # Embedding of position i
        def pos_emb_at(i):
            result = []
            for d in range(self.dim):
                if d%2 == 0:
                    emb_val = math.sin(i / (10000**(d/self.dim)))
                else:
                    emb_val = math.cos(i / (10000**((d-1)/self.dim)))
                result.append(emb_val)
            return result
        
        batch, seq_len = x.size()
        emb = []
        for i in range(seq_len):
            emb.append(pos_emb_at(i))
        
        emb = torch.tensor(emb)
        
        return emb.repeat(batch, seq_len, 1)    