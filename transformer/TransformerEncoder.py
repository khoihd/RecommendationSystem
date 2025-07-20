import torch.nn as nn
from PositionalWordEmbedding import PositionalWordEmbedding
from AddNorm import AddNorm
from Attention import Attention

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, att_dim=64, att_heads=8, ffn_dim=2048, layers=6):
        super().__init__()
        # Embedd input
        self.embedding = PositionalWordEmbedding(vocab_size, emb_dim)
        
        self.layers = layers
        self.encoding_layers = []    
        for _ in range(layers):
            self.encoding_layers.append([
                Attention(emb_dim, att_dim, att_heads),
                AddNorm(),
                nn.Linear(att_heads * att_dim, ffn_dim, bias=True),
                nn.ReLU(),
                nn.Linear(ffn_dim, att_heads * att_dim, bias=True),
                AddNorm()
            ])

    def forward(self, x):
        x = self.embedding(x)
        for encoder in self.encoding_layers:
            att_x = encoder[0](x)       # attention
            x = encoder[1](x, att_x)    # add_norm
            ffc_x = encoder[2](x)       # fcc
            ffc_x = encoder[3](ffc_x)   # relu
            ffc_x = encoder[4](ffc_x)   # fcc
            x = encoder[5](x, ffc_x)    # add_norm
        
        return x
