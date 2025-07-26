import torch.nn as nn
from . import AddNorm, Attention, PositionalWordEmbedding

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, attn_heads=8, 
                 ffn_dim=2048, layers=6, max_seq_len=50):
        assert emb_dim % attn_heads == 0, "emb_dim must be divisible by attn_heads"
        super().__init__()
        self.embedding = PositionalWordEmbedding(vocab_size, emb_dim, max_seq_len)
        self.encoding_layers = nn.ModuleList(
            [
                EncoderLayer(emb_dim, attn_heads, ffn_dim)
                for _ in range(layers)
            ]
        )

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.encoding_layers:
            x = layer(x)
        
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, attn_heads, ffn_dim):
        super().__init__()
        self.attention = Attention(emb_dim, attn_heads)
        self.addnorm1 = AddNorm(emb_dim)
        self.fcc = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ffn_dim, emb_dim, bias=True)
        )
        self.addnorm2 = AddNorm(emb_dim)
    
    def forward(self, x):
        attn_out = self.attention(x, x, x)
        x = self.addnorm1(x, attn_out)
        fcc_out = self.fcc(x)
        x = self.addnorm2(x, fcc_out)

        return x
