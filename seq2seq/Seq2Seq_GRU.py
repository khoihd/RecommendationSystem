import torch.nn as nn

class Seq2Seq_GRU(nn.Module):
    def __init__(self, input_dim, input_emb_dim, encoder_hidden_dim, 
                    output_dim, output_emb_dim, decoder_hidden_dim, layer=1, bidirectional=False):

        super().__init__()
        self.encoder_emb = nn.Embedding(input_dim, input_emb_dim)
        self.encoder = nn.GRU(input_emb_dim, encoder_hidden_dim, layer, bias=True, batch_first=True, bidirectional=bidirectional)
        
        if bidirectional:
            layer = layer * 2
        self.decoder_emb = nn.Embedding(output_dim, output_emb_dim)
        self.decoder = nn.GRU(output_emb_dim, decoder_hidden_dim, layer, bias=True, batch_first=True, bidirectional=False)
        self.fcc = nn.Linear(decoder_hidden_dim, output_dim)

    def forward(self, encoder_input, decoder_input):
        encoder_emb_input = self.encoder_emb(encoder_input)
        _, state = self.encoder(encoder_emb_input)
        decoder_emb_input = self.decoder_emb(decoder_input)
        decoder_output, decoder_state = self.decoder(decoder_emb_input, state)
        return self.fcc(decoder_output), decoder_state