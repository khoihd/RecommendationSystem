import math
import numpy as np
import torch
import sys
from PositionalWordEmbedding import PositionalWordEmbedding
from PytorchPositionalEncoding import PytorchPositionalEncoding

import unittest
unittest.TestLoader.sortTestMethodsUsing = None

# Test class
class TestPositionalEncoding(unittest.TestCase):    
    PRECISION = 1e-5
    def test_01_precision(self):
        datapoints = 100000
        np_values = np.arange(0, 2*np.pi, datapoints)
        torch_values = torch.arange(0, 2*torch.pi, datapoints)
        
        print("Running precision test for sine and cosine functions from torch and numpy...")
        assert torch.sin(torch_values).numpy() == np.sin(np_values)
        assert torch.cos(torch_values).numpy() == np.cos(np_values)
        print("Done")
    
    def test_02_positional_encoding(self):
        batch_size = 64
        seq_len = 100
        vocab_size, emb_dim = 1000, 512

        training_set = []
        for batch in range(batch_size):
            training_set.append(list(range(seq_len)))

        training_set = torch.tensor(training_set)

        positional_embedding = PositionalWordEmbedding(vocab_size, seq_len, emb_dim)
        emb = positional_embedding.positional_embeddding(training_set)

        pytorch_positional_embedding = PytorchPositionalEncoding(emb_dim, dropout=0, max_len=seq_len)
        pytorch_emb = pytorch_positional_embedding(training_set.transpose(0, 1))
        pytorch_emb = pytorch_emb.transpose(0, 1)

        print(f"Running test for positional embedding with batch_size={batch_size}, seq_len={seq_len}, emb_dim={emb_dim}...")
        print(f"pytorch_emb size:{pytorch_emb.size()}")
        print(f"emb size:{emb.size()}")
        for batch in range(batch_size):
            for i in range(seq_len):
                for d in range(emb_dim):
                    assert emb[batch][i][d] == pytorch_emb[batch][i][d]
                    if d%2 == 0:
                        emb_val = torch.sin(torch.tensor(i / 10000**(d/emb_dim))).item()
                    else:
                        emb_val = torch.cos(torch.tensor(i / 10000**((d-1)/emb_dim))).item()
                    
                    assert abs(emb[batch][i][d] - emb_val) < TestPositionalEncoding.PRECISION
                        

# Run tests if executed directly
if __name__ == '__main__':
    unittest.main()