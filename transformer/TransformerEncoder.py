import torch.nn as nn
from . import AddNorm, Attention, PositionalWordEmbedding

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, attn_dim=64, attn_heads=8, ffn_dim=2048, layers=6, max_seq_len=50):
        super().__init__()
        self.embedding = PositionalWordEmbedding(vocab_size, emb_dim, max_seq_len)
        self.layers = layers
        self.encoding_layers = nn.Sequential()

        for _ in range(layers):
            layer = nn.Sequential(
                Attention(emb_dim, attn_dim, attn_heads, mask=False),
                AddNorm(attn_heads * attn_dim),
                nn.Linear(attn_heads * attn_dim, ffn_dim, bias=True),
                nn.ReLU(),
                nn.Linear(ffn_dim, attn_heads * attn_dim, bias=True),
                AddNorm(attn_heads * attn_dim)
            )
            self.encoding_layers.append(layer)

    def forward(self, x):
        x = self.embedding(x)
        for encoder in self.encoding_layers:
            att_x = encoder[0](x, x, x) # attention
            x = encoder[1](x, att_x)    # add_norm
            ffc_x = encoder[2](x)       # fcc
            ffc_x = encoder[3](ffc_x)   # relu
            ffc_x = encoder[4](ffc_x)   # fcc
            x = encoder[5](x, ffc_x)    # add_norm
        
        return x
