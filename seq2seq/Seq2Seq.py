import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input):
        z = self.encoder(encoder_input)
        decoder_output, decoder_state_layer = self.decoder(decoder_input, z)
        return decoder_output, decoder_state_layer