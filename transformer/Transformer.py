import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerEncoder, TransformerDecoder


class Transformer(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        emb_dim=512,
        attn_dim=64,
        attn_heads=8,
        ffn_dim=2048,
        layers=6,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            input_vocab_size, emb_dim=emb_dim, attn_dim=attn_dim,
            attn_heads=attn_heads, ffn_dim=ffn_dim, layers=layers,
        )
        self.decoder = TransformerDecoder(
            output_vocab_size, emb_dim=emb_dim, attn_dim=attn_dim,
            attn_heads=attn_heads, ffn_dim=ffn_dim, layers=layers,
        )

    def forward(self, x, y):
        encoder_output = self.encoder(x)
        return self.decoder(y, encoder_output)

    def translate(self, sequence_tokens, en_idx_token_dict, device, eos, sos_idx=2, max_output_len=100):
        self.eval()
        sequence_tokens_batch = sequence_tokens.unsqueeze(0)
        sequence_output = []
        with torch.no_grad():
            # encoder_input_emb = self.encoder_emb(sequence_tokens_batch)
            # encoder_sequence_state, state = self.encoder(encoder_input_emb)
            encoder_state = self.encoder(sequence_tokens_batch)
            word_idx = sos_idx
            for _ in range(max_output_len):
                decoder_input = torch.tensor([[word_idx]]).to(device)
                word_output = self.decoder(decoder_input, encoder_state)
                word_output = F.softmax(word_output, dim=-1)                
                word_idx = torch.argmax(word_output, dim=-1).item()
                word_token = en_idx_token_dict[word_idx]
                sequence_output.append(word_token)
                if word_token == eos:
                    break

        return sequence_output
