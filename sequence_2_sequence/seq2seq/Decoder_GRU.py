import torch.nn as nn

class Decoder_GRU(nn.Module):
    def __init__(self, output_dim, embedding_dim, rnn_hidden_dim, rnn_num_layers, bidirectional):
        super().__init__()
        if bidirectional:
            rnn_num_layers = rnn_num_layers * 2
            
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.decoder = nn.GRU(embedding_dim, rnn_hidden_dim, rnn_num_layers, batch_first=True, bias=True)
        self.fc = nn.Linear(rnn_hidden_dim, output_dim)

    def forward(self, x, latent):
        x = self.embedding(x)
        # hidden state at the last layer for every word in the sequence:
        #       batch, sequence, hidden_dim
        # final hidden state at every layer
        #       layer, batch, hidden_dim
        state_sequence, state_layer = self.decoder(x, latent)
        return self.fc(state_sequence), state_layer