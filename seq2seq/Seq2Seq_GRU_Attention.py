import torch.nn as nn

class Seq2Seq_GRU_Attention(nn.Module):
    # Encoder: Input -> Embedded -> GRU (hidden, layers, bidirectional)
    # Decoder: Ouput -> Embedded -> GRU (hidden, layers)

    # Attention
    # Attention scores = multiply every sequence state with the current decoder hidden state
    # Softmax on the multiplication result => attention weights
    # 
    def __init__(self, input_dim, input_emb_dim, output_dim, output_emb_dim,
                        hidden_dim, layer, bidirectional):
        self.encoder_imb = nn.Embedding(input_dim, input_emb_dim)
        self.encoder = nn.GRU(input_emb_dim, hidden_dim, num_layers=layer, batch_first=True, bidirectional=bidirectional)
        self.decoder_imb = nn.Embedding(output_dim, output_emb_dim)
        # TODO: Update the number of layers based on bidirectional
        self.decoder = nn.GRU(output_emb_dim, hidden_dim, num_layers=layer, batch_first=True)
        # output from decoder_hidden_dim directly to output dim
        self.softmax = nn.Softmax()
        self.fcc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, encoder_input, decoder_input):
        encoder_imb = self.encoder_imb(encoder_input)
        encoder_sequence_state, encoder_layer_state = self.encoder(encoder_imb)
        decoder_imb = self.decoder_imb(decoder_input)
        decoder_sequence_state, decoder_layer_state = self.decoder(decoder_imb, encoder_layer_state)
        # Attention scores: Multiply encoder_sequence_state with decoder_sequence_state
        
        # encoder_sequence_state: batch x input_seq_len x hidden_dim
        # decoder_layer_state: batch x output_seq_len x hidden_dim
        encoder_sequence_state @ decoder_layer_state

        # Attention weights: Then compute softmax at every element from the decoder sequence state
        
        # Multiply the attention weights with encoder_sequence_state, and combined into a vector of size decoder_hidden_dim
        
        # Then concat the attention value with decoder_sequence_state
        
        output = self.fcc(decoder_sequence_state)

        return output, decoder_sequence_state, decoder_layer_state
