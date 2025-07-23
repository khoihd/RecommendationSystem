import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


DOT_PRODUCT = "DOT_PRODUCT"
SCALED_DOT_PRODUCT = "SCALED_DOT_PRODUCT"
GENERALIZED_DOT_PRODUCT = "GENERALIZED_DOT_PRODUCT"

LOCAL_MONOTONIC_ALIGN = "LOCAL_MONOTONIC"
LOCAL_PREDICTIVE_ALIGN = "LOCAL_PREDICTIVE_ALIGN"

class Seq2Seq_GRU_Attention(nn.Module):
    # Encoder: Input -> Embedded -> GRU (hidden, layers, bidirectional)
    # Decoder: Ouput -> Embedded -> GRU (hidden, layers) 

    def __init__(self, input_dim, input_emb_dim, output_dim, output_emb_dim,
                        hidden_dim, layer, attention_type, bidirectional):
        super().__init__()
        self.encoder_emb = nn.Embedding(input_dim, input_emb_dim)
        self.decoder_emb = nn.Embedding(output_dim, output_emb_dim)
        self.encoder = nn.GRU(input_emb_dim, hidden_dim, num_layers=layer, batch_first=True, bidirectional=bidirectional)
        
        if not bidirectional:
            self.decoder = nn.GRU(output_emb_dim, hidden_dim, num_layers=layer, batch_first=True)
            self.encoder_attn = None
        else:
            self.decoder = nn.GRU(output_emb_dim, hidden_dim, num_layers=layer*2, batch_first=True)
            self.encoder_attn = nn.Linear(hidden_dim*2, hidden_dim)
        
        self.general_attn_matrix = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.attention_type = attention_type
        self.fcc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, encoder_input, decoder_input):
        encoder_emb = self.encoder_emb(encoder_input)
        encoder_sequence_state, encoder_layer_state = self.encoder(encoder_emb)
        decoder_emb = self.decoder_emb(decoder_input)
        decoder_sequence_state, _ = self.decoder(decoder_emb, encoder_layer_state)
        
        # Attention scores: Multiply decoder_sequence_state (query) with encoder_sequence_state (key)
        # key: encoder_sequence_state: batch x input_seq_len x hidden_dim
        # query: decoder_sequence_state: batch x output_seq_len x hidden_dim
        # attention_scores, attention_weights: batch x output_seq_len x input_seq_len
        # attention_value: batch x output_seq_len x hidden_dim
        #   attention_value = attention_weights @ encoder_sequence_state
        
        attention_func = self.get_attention_fn()
        concat = attention_func(query=decoder_sequence_state, key=encoder_sequence_state)
        output = self.fcc(concat)
        return output, decoder_sequence_state

    def get_attention_fn(self):
        def dot_product_attn(query, key):
            if self.encoder_attn is not None:
                key = self.encoder_attn(key)
            
            if self.attention_type == DOT_PRODUCT:
                attention_scores = query @ key.permute(0, 2, 1)
            elif self.attention_type == SCALED_DOT_PRODUCT:
                attention_scores = query @ key.permute(0, 2, 1) / np.sqrt(key.size()[-1])
            elif self.attention_type == GENERALIZED_DOT_PRODUCT:
                attention_scores = query @ self.general_attn_matrix @ key.permute(0, 2, 1)
            
            attention_weights = F.softmax(attention_scores, dim=-1)
            context_vector = attention_weights @ key
            concat = torch.concat((query, context_vector), dim=-1)

            return concat
    
        return dot_product_attn
    
    def translate(self, sequence_tokens, en_idx_token_dict, device, eos, sos_idx=2, max_output_len=100):
        self.eval()
        sequence_tokens_batch = sequence_tokens.unsqueeze(0)
        attention_fn = self.get_attention_fn()
        sequence_output = []
        with torch.no_grad():
            encoder_input_emb = self.encoder_emb(sequence_tokens_batch)
            encoder_sequence_state, state = self.encoder(encoder_input_emb)
            word_idx = sos_idx
            for _ in range(max_output_len):
                word_encoder = torch.tensor([[word_idx]]).to(device)
                word_emb = self.decoder_emb(word_encoder)
                
                decoder_sequence_state, state = self.decoder(word_emb, state)
                concat = attention_fn(query=decoder_sequence_state, key=encoder_sequence_state)
                word_output = self.fcc(concat)

                # attention_scores = decoder_sequence_state @ encoder_sequence_attn.permute(0, 2, 1)
                # attention_weights = F.softmax(attention_scores, dim=-1)
                # attention_values = attention_weights @ encoder_sequence_attn
                # concat = torch.concat((decoder_sequence_state, attention_values), dim=-1)
                # word_output = self.fcc(concat)
                
                word_output = F.softmax(word_output, dim=-1)
                word_idx = torch.argmax(word_output, dim=-1).item()
                word_token = en_idx_token_dict[word_idx]
                sequence_output.append(word_token)
                if word_token == eos:
                    break
                
        return sequence_output
