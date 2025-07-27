import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerEncoder, TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size,
        emb_dim=512, attn_heads=8, ffn_dim=2048, layers=6, max_seq_len=50
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            input_vocab_size, emb_dim=emb_dim, attn_heads=attn_heads,
            ffn_dim=ffn_dim, layers=layers, max_seq_len=max_seq_len
        )
        self.decoder = TransformerDecoder(
            output_vocab_size, emb_dim=emb_dim, attn_heads=attn_heads,
            ffn_dim=ffn_dim, layers=layers, max_seq_len=max_seq_len
        )

    def forward(self, x, y):
        enc_x = self.encoder(x)
        return self.decoder(y, enc_x)

    def translate(self, sequence_tokens, en_idx_token_dict, device, eos, sos_idx=2, max_output_len=100):
        self.eval()
        sequence_tokens_batch = sequence_tokens.unsqueeze(0)
        decoder_input = torch.tensor([sos_idx]).unsqueeze(0).to(device)
        sequence_output = []
        with torch.no_grad():
            encoder_state = self.encoder(sequence_tokens_batch)
            # print("sequence_tokens_batch", sequence_tokens_batch)
            # print("encoder_state", encoder_state)
            for _ in range(max_output_len):
                sentence_output = self.decoder(decoder_input, encoder_state)
                sentence_output = sentence_output[0,-1,:]
                next_token_idx = torch.argmax(sentence_output, dim=-1).item()                
                decoder_input = torch.concat((decoder_input,
                                             torch.tensor([next_token_idx]).unsqueeze(0)),
                                             dim=-1)

                next_token = en_idx_token_dict[next_token_idx]
                sequence_output.append(next_token)
                if next_token == eos:
                    break

        return sequence_output
