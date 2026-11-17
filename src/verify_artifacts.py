"""Phase 4: Fresh-session verification of serialized vectorizer.pkl and model.pkl."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Tuple

from src.preprocess import transform_text


def load_artifacts(
    vectorizer_path: Path = Path("vectorizer.pkl"),
    model_path: Path = Path("model.pkl"),
) -> Tuple[object, object]:
    """Load serialized TF-IDF vectorizer and MultinomialNB model."""
    if not vectorizer_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "vectorizer.pkl and/or model.pkl not found. Run `python -m src.train` first."
        )
    with vectorizer_path.open("rb") as vf:
        vectorizer = pickle.load(vf)
    with model_path.open("rb") as mf:
        model = pickle.load(mf)
    return vectorizer, model


def verify_predictions() -> None:
    """Run fresh-session predictions and check sub-second inference latency."""
    vectorizer, model = load_artifacts()

    samples = [
        (
            "URGENT! You have won a $1,000 Walmart gift card. Click http://bit.ly/claim-now to claim immediately!",
            "Spam",
        ),
        (
            "Hey, are we still meeting at 4 PM near the library to review the project slides?",
            "Not Spam",
        ),
    ]

    for text, expected in samples:
        start = time.perf_counter()
        transformed = transform_text(text)
        vectorized = vectorizer.transform([transformed])
        pred = int(model.predict(vectorized)[0])
        proba = float(model.predict_proba(vectorized)[0][1])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        label = "Spam" if pred == 1 else "Not Spam"
        print(
            f"Expected={expected:8s} | Predicted={label:8s} | "
            f"P(spam)={proba:.4f} | Latency={elapsed_ms:.3f} ms"
        )
        assert elapsed_ms < 1000.0, f"Inference exceeded 1s: {elapsed_ms:.3f} ms"
        assert label == expected, f"Unexpected classification for sample: {text}"


if __name__ == "__main__":
    verify_predictions()
    print("Fresh-session artifact verification passed.")
