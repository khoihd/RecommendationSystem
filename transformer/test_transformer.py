import math
import unittest
import torch
from PositionalWordEmbedding import PositionalWordEmbedding

# Test class
class TestPositionalEncoding(unittest.TestCase):
    def test_positional_encoding(self):
        batch_size = 64
        seq_len = 100
        vocab_size, emb_dim = 1000, 512

        training_set = []
        for batch in range(batch_size):
            training_set.append(list(range(seq_len)))

        training_set = torch.tensor(training_set)

        positional_embedding = PositionalWordEmbedding(vocab_size, emb_dim)
        emb = positional_embedding.positional_embeddding(training_set)

        print(f"Running test for positional embedding with batch_size={batch_size}, seq_len={seq_len}, emb_dim={emb_dim}...")
        for batch in range(batch_size):
            for i in range(seq_len):
                for d in range(emb_dim):
                    if d%2 == 0:
                        # assert abs(emb[batch][i][d] - math.sin(i / 10000**(d/emb_dim))) < sys.float_info.epsilon
                        assert emb[batch][i][d] == math.sin(i / 10000**(d/emb_dim))
                    else:
                        # assert abs(emb[batch][i][d] - math.cos(i / 10000**((d-1)/emb_dim))) < sys.float_info.epsilon
                        assert emb[batch][i][d] == math.cos(i / 10000**((d-1)/emb_dim))

# Run tests if executed directly
if __name__ == '__main__':
    unittest.main()