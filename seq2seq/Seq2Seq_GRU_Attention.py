import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2Seq_GRU_Attention(nn.Module):
    # Encoder: Input -> Embedded -> GRU (hidden, layers, bidirectional)
    # Decoder: Ouput -> Embedded -> GRU (hidden, layers)

    # Attention
    # Attention scores = multiply every sequence state with the current decoder hidden state
    # Softmax on the multiplication result => attention weights

    DOT_PRODUCT_ATTENTION = 'DOT_PRODUCT_ATTENTION'
    
    def __init__(self, input_dim, input_emb_dim, output_dim, output_emb_dim,
                        hidden_dim, layer, attention_type, bidirectional):
        super().__init__()
        self.encoder_emb = nn.Embedding(input_dim, input_emb_dim)
        self.encoder = nn.GRU(input_emb_dim, hidden_dim, num_layers=layer, batch_first=True, bidirectional=bidirectional)
        self.decoder_emb = nn.Embedding(output_dim, output_emb_dim)
        if bidirectional:
            hidden_dim = hidden_dim * 2
        
        self.decoder = nn.GRU(output_emb_dim, hidden_dim, num_layers=layer, batch_first=True)
        self.fcc = nn.Linear(2 * hidden_dim, output_dim)
        self.attention_type = attention_type

    def forward(self, encoder_input, decoder_input):
        encoder_emb = self.encoder_emb(encoder_input)
        encoder_sequence_state, encoder_layer_state = self.encoder(encoder_emb)
        decoder_emb = self.decoder_emb(decoder_input)
        decoder_sequence_state, _ = self.decoder(decoder_emb, encoder_layer_state)
        # Attention scores: Multiply encoder_sequence_state with decoder_sequence_state
        
        #   Get the last layer of encoder_layer_state: encoder_layer_state[:,-1,:]
        # encoder_sequence_state: batch x input_seq_len x hidden_dim
        # decoder_sequence_state: batch x output_seq_len x hidden_dim
        # appended_decoder_state = encoder_sequence_state[:,-1:,:] concat decoder_sequence_state
        # attention_scores, attention_weights: batch x output_seq_len x input_seq_len
        # attention_value: batch x output_seq_len x hidden_dim
        #   attention_value = attention_weights @ encoder_sequence_state
        attention_func = self.get_attention_fn(self.attention_type)
        appended_decoder_state = torch.concat(
            (encoder_sequence_state[:,-1:,:], decoder_sequence_state), dim=1)

        concat = attention_func(query=encoder_sequence_state, value=appended_decoder_state[:,:-1,:])
        output = self.fcc(concat)
        return output, decoder_sequence_state

    def get_attention_fn(self, attention_type):
        def dot_product_attn(query, value):
            attention_scores = value @ query.permute(0, 2, 1)
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_values = attention_weights @ query
            concat = torch.concat((value, attention_values), dim=-1)

            return concat
        
        if attention_type == Seq2Seq_GRU_Attention.DOT_PRODUCT_ATTENTION:
            return dot_product_attn
    
    def translate(self, sequence_tokens, en_idx_token_dict, device, eos, sos_idx=2, max_output_len=100):
        self.eval()
        sequence_tokens_batch = sequence_tokens.unsqueeze(0)
        
        sequence_output = []
        with torch.no_grad():
            encoder_input_emb = self.encoder_emb(sequence_tokens_batch)
            encoder_sequence_state, state = self.encoder(encoder_input_emb)
            word_idx = sos_idx
            for _ in range(max_output_len):
                word_encoder = torch.tensor([[word_idx]]).to(device)
                word_emb = self.decoder_emb(word_encoder)
                
                decoder_sequence_state, state = self.decoder(word_emb, state)
                attention_scores = decoder_sequence_state @ encoder_sequence_state.permute(0, 2, 1)
                attention_weights = F.softmax(attention_scores, dim=-1)
                attention_values = attention_weights @ encoder_sequence_state
                
                concat = torch.concat((decoder_sequence_state, attention_values), dim=-1)
                word_output = self.fcc(concat)
                
                word_idx = torch.argmax(word_output, dim=-1).item()
                word_token = en_idx_token_dict[word_idx]
                sequence_output.append(word_token)
                if word_token == eos:
                    break
                
        return sequence_output
