"""PyTorch Bidirectional LSTM architecture, vocabulary encoder, and inference helper."""

from __future__ import annotations

import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.preprocess import transform_text

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class Vocabulary:
    """Deterministic token-to-index vocabulary for LSTM sequence encoding."""

    def __init__(self, max_vocab_size: int = 8000, min_freq: int = 2) -> None:
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.token_to_idx: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    def fit(self, texts: Iterable[str]) -> "Vocabulary":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(text.split())

        most_common = [
            token
            for token, freq in counts.most_common(self.max_vocab_size - 2)
            if freq >= self.min_freq
        ]
        for token in most_common:
            if token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.token_to_idx)
        return self

    def encode(self, text: str, max_len: int = 100) -> List[int]:
        tokens = text.split()[:max_len]
        unk_idx = self.token_to_idx[UNK_TOKEN]
        ids = [self.token_to_idx.get(tok, unk_idx) for tok in tokens]
        if len(ids) < max_len:
            ids.extend([0] * (max_len - len(ids)))
        return ids

    def __len__(self) -> int:
        return len(self.token_to_idx)


class SpamBiLSTM(nn.Module):
    """Bidirectional LSTM classifier that also exposes a normalized 128-dim text embedding."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # Combine masked mean + max pooling into a compact 128-dim semantic embedding
        self.proj = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_features(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return a 128-dim masked mean+max pooled BiLSTM representation."""
        non_pad = token_ids != 0
        mask = non_pad.unsqueeze(-1).float()  # (B, L, 1)
        embedded = self.dropout(self.embedding(token_ids))
        outputs, _ = self.lstm(embedded)  # (B, L, hidden_dim * 2)

        summed = (outputs * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        mean_pooled = summed / lengths

        masked_for_max = outputs.masked_fill(~non_pad.unsqueeze(-1), -1e9)
        max_pooled = masked_for_max.max(dim=1).values
        has_tokens = non_pad.any(dim=1, keepdim=True)
        max_pooled = torch.where(has_tokens, max_pooled, torch.zeros_like(max_pooled))

        combined = torch.cat([mean_pooled, max_pooled], dim=-1)
        return self.norm(torch.tanh(self.proj(combined)))

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode_features(token_ids)
        logits = self.classifier(self.dropout(features)).squeeze(-1)
        return logits, features


class LstmSpamPredictor:
    """Wrapper for loading serialized BiLSTM artifacts and running sub-second inference."""

    def __init__(
        self,
        model_path: Path = Path("lstm_model.pt"),
        vocab_path: Path = Path("lstm_vocab.pkl"),
    ) -> None:
        if not model_path.exists() or not vocab_path.exists():
            raise FileNotFoundError(
                "Missing lstm_model.pt or lstm_vocab.pkl. Run `python -m src.train_lstm` first."
            )

        with vocab_path.open("rb") as vf:
            meta: Dict[str, Any] = pickle.load(vf)

        self.vocab: Vocabulary = meta["vocab"]
        self.max_len: int = int(meta["max_len"])
        self.threshold: float = float(meta.get("threshold", 0.5))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model = SpamBiLSTM(
            vocab_size=len(self.vocab),
            embed_dim=int(meta.get("embed_dim", 128)),
            hidden_dim=int(meta.get("hidden_dim", 64)),
            num_layers=int(meta.get("num_layers", 2)),
            dropout=float(meta.get("dropout", 0.25)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify text and return prediction, probability, 128-dim embedding, and timing."""
        t0 = time.perf_counter()
        transformed = transform_text(text)
        encoded = self.vocab.encode(transformed, max_len=self.max_len)
        tensor = torch.tensor([encoded], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits, features = self.model(tensor)
            spam_prob = float(torch.sigmoid(logits)[0].item())
            embedding = features[0].cpu().numpy().astype(np.float32)

        # L2-normalize embedding for cosine similarity in pgvector
        norm = float(np.linalg.norm(embedding))
        if norm > 0.0:
            embedding = embedding / norm

        pred = 1 if spam_prob >= self.threshold else 0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "prediction": pred,
            "label": "Spam" if pred == 1 else "Not Spam",
            "spam_probability": spam_prob,
            "ham_probability": 1.0 - spam_prob,
            "threshold": self.threshold,
            "transformed_text": transformed,
            "embedding": embedding,
            "inference_ms": elapsed_ms,
        }
