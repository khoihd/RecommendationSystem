import torch.nn as nn
from transformer import PositionalWordEmbedding, Attention, AddNorm

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, attn_dim=64, attn_heads=8, ffn_dim=2048, layers=6, max_seq_len=50):
        super().__init__()
        self.embedding = PositionalWordEmbedding(vocab_size, emb_dim, max_seq_len)
        self.layers = layers
        self.decoding_layers = nn.Sequential()
        for _ in range(layers):
            layer = nn.Sequential(
                Attention(emb_dim, attn_dim, attn_heads, mask=True),
                AddNorm(attn_heads * attn_dim),
                Attention(emb_dim, attn_dim, attn_heads, mask=False),
                AddNorm(attn_heads * attn_dim),
                nn.Linear(attn_heads * attn_dim, ffn_dim, bias=True),
                nn.ReLU(),
                nn.Linear(ffn_dim, attn_heads * attn_dim, bias=True)
            )
        self.fcc = nn.Linear(attn_heads * attn_dim, vocab_size, bias=True)

    def forward(self, x, enc_x):
        x = self.embedding(x)
        for layer in self.decoding_layers:
            attn_x = layer[0](x, x, x)                      # masked self-attention
            attn_x = layer[1](x, attn_x)                    # x + attn_n
            cross_attn_x = layer[2](attn_x, enc_x, enc_x)   # cross attention
            pre_fcc_x = layer[3](attn_x, cross_attn_x)      # attn_n + cross_attn_x
            fcc_x = layer[4](pre_fcc_x)                     # fcc
            fcc_x = layer[5](fcc_x)                         # relu
            fcc_x = layer[6](fcc_x)                         # fcc
            x = layer[7](pre_fcc_x, fcc_x)                  # pre_fcc_x + fcc_x

        return self.fcc(x)                                  # final fcc