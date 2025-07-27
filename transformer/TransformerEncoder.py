import torch.nn as nn
from . import Attention, PositionalWordEmbedding

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, attn_heads=8, 
                 ffn_dim=2048, layers=6, max_seq_len=50, pre_norm=False):
        assert emb_dim % attn_heads == 0, "emb_dim must be divisible by attn_heads"
        super().__init__()
        self.embedding = PositionalWordEmbedding(vocab_size, emb_dim, max_seq_len)
        self.encoding_layers = nn.ModuleList(
            [
                EncoderLayer(emb_dim, attn_heads, ffn_dim, pre_norm)
                for _ in range(layers)
            ]
        )

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.encoding_layers:
            x = layer(x)
        
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, attn_heads, ffn_dim, pre_norm):
        super().__init__()
        self.pre_norm = pre_norm
        self.attention = Attention(emb_dim, attn_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.fcc = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ffn_dim, emb_dim, bias=True)
        )
        self.norm2 = nn.LayerNorm(emb_dim)
    
    def forward(self, x):
        if not self.pre_norm:
            x = self.norm1(x + self.attention(x, x, x))
            x = self.norm2(x + self.fcc(x))
        
            return x
        
        if self.pre_norm:
            x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
            x = x + self.fcc(self.norm2(x))

            return x

