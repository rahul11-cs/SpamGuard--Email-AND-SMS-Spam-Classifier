"""Unit tests for text preprocessing pipeline."""

import unittest

from src.preprocess import transform_text


class TestPreprocess(unittest.TestCase):
    def test_lowercasing_and_url_stripping(self) -> None:
        raw = "Check THIS out: https://spam.example.com/win?id=123 NOW!!!"
        out = transform_text(raw)
        self.assertNotIn("https", out)
        self.assertNotIn("example", out)
        self.assertEqual(out, out.lower())

    def test_punctuation_and_non_alnum_removed(self) -> None:
        raw = "Free$$$ cash!!! Call #12345 @now."
        out = transform_text(raw)
        for ch in ["$", "!", "#", "@", "."]:
            self.assertNotIn(ch, out)
        self.assertIn("free", out)
        self.assertIn("cash", out)

    def test_stopwords_and_stemming(self) -> None:
        raw = "He is running and playing games with the friends"
        out = transform_text(raw)
        tokens = out.split()
        self.assertNotIn("is", tokens)
        self.assertNotIn("the", tokens)
        self.assertIn("run", tokens)
        self.assertIn("play", tokens)
        self.assertIn("game", tokens)

    def test_empty_input(self) -> None:
        self.assertEqual(transform_text(""), "")
        self.assertEqual(transform_text("!!! ???"), "")


if __name__ == "__main__":
    unittest.main()
