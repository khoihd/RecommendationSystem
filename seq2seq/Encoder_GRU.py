import torch.nn as nn

class Encoder_GRU(nn.Module):
    def __init__(self, input_dim, embedding_dim, rnn_hidden_dim, rnn_num_layers, bidirectional):
        super().__init__()
        # 1 layer Embedding
        # 2 layers GRU
        # the latent space is the same as the hidden space of the last layer of the GRU
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, rnn_hidden_dim, rnn_num_layers, bidirectional=bidirectional, batch_first=True, bias=True)

    def forward(self, x):
        # hidden state at the last layer for every word in the sequence:
        #       batch, sequence, hidden_dim
        # final hidden state at every layer
        #       layer, batch, hidden_dim
        x = self.embedding(x)
        _, state_layer = self.encoder(x)
        return state_layer