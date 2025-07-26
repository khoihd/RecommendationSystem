import torch.nn as nn
from transformer import PositionalWordEmbedding, Attention, AddNorm

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, attn_dim=64, attn_heads=8, 
                 ffn_dim=2048, layers=6, max_seq_len=50):
        assert emb_dim % attn_heads == 0, "emb_dim must be divisible by attn_heads"
        super().__init__()
        self.embedding = PositionalWordEmbedding(vocab_size, emb_dim, max_seq_len)
        self.layers = layers
        self.decoding_layers = nn.ModuleList(
            [
                DecoderLayer(emb_dim, attn_heads, ffn_dim)
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
    def __init__(self, emb_dim, attn_heads, ffn_dim):
        super().__init__()
        self.mask_attention = Attention(emb_dim, attn_heads)
        self.addnorm1 = AddNorm(emb_dim)
        self.cross_attention = Attention(emb_dim, attn_heads)
        self.addnorm2 = AddNorm(emb_dim)
        self.fcc = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ffn_dim, emb_dim, bias=True)
        )
        self.addnorm3 = AddNorm(emb_dim)
    
    def forward(self, x, enc_x):
        mask_attn_out = self.mask_attention(x, x, x, mask=True)
        x = self.addnorm1(x, mask_attn_out)
        cross_attn_out = self.cross_attention(x, enc_x, enc_x)
        x = self.addnorm2(x, cross_attn_out)
        fcc_out = self.fcc(x)
        x = self.addnorm3(x, fcc_out)

        return x