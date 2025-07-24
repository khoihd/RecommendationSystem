import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerEncoder, TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size,
        emb_dim=512, attn_dim=64, attn_heads=8, ffn_dim=2048, layers=6, max_seq_len=30
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            input_vocab_size, emb_dim=emb_dim, attn_dim=attn_dim,
            attn_heads=attn_heads, ffn_dim=ffn_dim, layers=layers, 
            max_seq_len=max_seq_len
        )
        self.decoder = TransformerDecoder(
            output_vocab_size, emb_dim=emb_dim, attn_dim=attn_dim,
            attn_heads=attn_heads, ffn_dim=ffn_dim, layers=layers, 
            max_seq_len=max_seq_len
        )

    def forward(self, x, y):
        encoder_output = self.encoder(x)
        return self.decoder(y, encoder_output)

    def translate(self, sequence_tokens, en_idx_token_dict, device, eos, sos_idx=2, max_output_len=100):
        self.eval()
        sequence_tokens_batch = sequence_tokens.unsqueeze(0)
        decoder_input = [[word_idx]]
        sequence_output = []
        with torch.no_grad():
            # encoder_input_emb = self.encoder_emb(sequence_tokens_batch)
            # encoder_sequence_state, state = self.encoder(encoder_input_emb)
            encoder_state = self.encoder(sequence_tokens_batch)
            word_idx = sos_idx
            for _ in range(max_output_len):
                decoder_input = torch.tensor().to(device)
                word_output = self.decoder(decoder_input, encoder_state)
                word_output = F.softmax(word_output, dim=-1)                
                word_idx = torch.argmax(word_output, dim=-1).item()
                decoder_input[0].append(word_idx)
                word_token = en_idx_token_dict[word_idx]
                sequence_output.append(word_token)
                if word_token == eos:
                    break

        return sequence_output
