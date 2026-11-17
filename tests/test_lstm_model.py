"""Unit tests for BiLSTM vocabulary and forward pass dimensions."""

import unittest

import torch

from src.lstm_model import SpamBiLSTM, Vocabulary


class TestLstmArchitecture(unittest.TestCase):
    def test_vocabulary_fit_and_encode(self) -> None:
        vocab = Vocabulary(max_vocab_size=100, min_freq=1).fit(
            ["win free prize", "meet team library"]
        )
        encoded = vocab.encode("win free unknownword", max_len=6)
        self.assertEqual(len(encoded), 6)
        self.assertEqual(encoded[2], 1)  # <UNK>
        self.assertEqual(encoded[3:], [0, 0, 0])  # <PAD>

    def test_bilstm_output_and_128d_embedding(self) -> None:
        model = SpamBiLSTM(vocab_size=50, embed_dim=32, hidden_dim=64, num_layers=2)
        sample_ids = torch.tensor([[2, 3, 4, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.long)
        logits, features = model(sample_ids)
        self.assertEqual(logits.shape, (2,))
        self.assertEqual(features.shape, (2, 128))


if __name__ == "__main__":
    unittest.main()
