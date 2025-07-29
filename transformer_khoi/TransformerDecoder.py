import torch.nn as nn
from transformer import PositionalWordEmbedding, Attention

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, attn_heads=8, 
                 ffn_dim=2048, layers=6, max_seq_len=50, pre_norm=False):
        assert emb_dim % attn_heads == 0, "emb_dim must be divisible by attn_heads"
        super().__init__()
        self.embedding = PositionalWordEmbedding(vocab_size, emb_dim, max_seq_len)
        self.layers = layers
        self.decoding_layers = nn.ModuleList(
            [
                DecoderLayer(emb_dim, attn_heads, ffn_dim, pre_norm)
                for _ in range(layers)
            ]
        )
        self.final_fcc = nn.Linear(emb_dim, vocab_size, bias=True)

    def forward(self, x, enc_x):
        x = self.embedding(x)
        for layer in self.decoding_layers:
            x = layer(x, enc_x)

        return self.final_fcc(x)
    

class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, attn_heads, ffn_dim, pre_norm):
        super().__init__()
        self.pre_norm = pre_norm
        self.mask_attention = Attention(emb_dim, attn_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.cross_attention = Attention(emb_dim, attn_heads)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.fcc = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ffn_dim, emb_dim, bias=True)
        )
        self.norm3 = nn.LayerNorm(emb_dim)
    
    def forward(self, x, enc_x):
        if not self.pre_norm:
            x = self.norm1(x + self.mask_attention(x, x, x, mask=True))
            x = self.norm2(x + self.cross_attention(x, enc_x, enc_x))
            x = self.norm3(x + self.fcc(x))

            return x

        if self.pre_norm:
            x = x + self.mask_attention(self.norm1(x), self.norm1(x), self.norm1(x), mask=True)
            x = x + self.cross_attention(self.norm2(x), self.norm2(enc_x), self.norm2(enc_x))
            x = x + self.fcc(self.norm3(x))

            return x