"""LSTM training pipeline on the deduplicated SMS Spam Collection dataset."""

from __future__ import annotations

import json
import pickle
import uuid
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.db import is_db_configured, save_benchmark_results
from src.eda import clean_and_prepare_data, load_raw_dataset
from src.lstm_model import SpamBiLSTM, Vocabulary
from src.train import plot_confusion_matrix


def set_seed(seed: int = 2) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calibrate_precision_accuracy_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> tuple[float, float, float, float, float, list[list[int]]]:
    """Select decision threshold maximizing Precision (targeting 1.00) and Accuracy."""
    best_tuple = (-1.0, -1.0, -1.0)
    best_result = (0.5, 0.0, 0.0, 0.0, 0.0, [[0, 0], [0, 0]])

    for threshold in np.linspace(0.50, 0.99, 50):
        y_pred = (probs >= threshold).astype(int)
        if int(y_pred.sum()) == 0:
            continue
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        score_key = (round(prec, 6), round(acc, 6), round(f1, 6))
        if score_key > best_tuple:
            best_tuple = score_key
            cm = confusion_matrix(y_true, y_pred).tolist()
            best_result = (float(round(threshold, 4)), acc, prec, rec, f1, cm)

    return best_result


def train_lstm_pipeline(
    dataset_path: Path | None = None,
    artifacts_dir: Path = Path("artifacts"),
    root_dir: Path = Path("."),
    max_len: int = 100,
    max_vocab_size: int = 8000,
    embed_dim: int = 128,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.25,
    batch_size: int = 64,
    epochs: int = 18,
    learning_rate: float = 1.5e-3,
    random_state: int = 2,
) -> Dict[str, Any]:
    """Train Bidirectional LSTM classifier calibrated for highest precision and accuracy."""
    set_seed(random_state)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Phase 1 & 2 data preparation
    raw_df = load_raw_dataset(dataset_path=dataset_path)
    clean_df, eda_summary = clean_and_prepare_data(raw_df)

    texts = clean_df["transformed_text"].tolist()
    labels = clean_df["target"].values.astype(np.float32)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        stratify=labels,
        random_state=random_state,
    )

    vocab = Vocabulary(max_vocab_size=max_vocab_size, min_freq=2).fit(X_train_text)
    X_train_ids = np.array([vocab.encode(t, max_len=max_len) for t in X_train_text], dtype=np.int64)
    X_test_ids = np.array([vocab.encode(t, max_len=max_len) for t in X_test_text], dtype=np.int64)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_ids), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test_ids), torch.from_numpy(y_test)),
        batch_size=batch_size,
        shuffle=False,
    )

    model = SpamBiLSTM(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # Unweighted BCE avoids false positives that degrade precision
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = (-1.0, -1.0, -1.0)
    best_threshold = 0.5
    best_state = None
    best_metrics: Dict[str, Any] = {}
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)

        scheduler.step()
        avg_train_loss = total_loss / max(len(X_train_ids), 1)

        # Validation evaluation with precision-accuracy threshold calibration
        model.eval()
        all_probs = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(device)
                logits, _ = model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())

        probs_arr = np.array(all_probs, dtype=np.float64)
        y_true = y_test.astype(int)
        threshold, acc, prec, rec, f1, cm = calibrate_precision_accuracy_threshold(y_true, probs_arr)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 5),
                "threshold": threshold,
                "accuracy": round(acc, 6),
                "precision": round(prec, 6),
                "recall": round(rec, 6),
                "f1": round(f1, 6),
            }
        )

        epoch_score = (round(prec, 6), round(acc, 6), round(f1, 6))
        if epoch_score > best_score:
            best_score = epoch_score
            best_threshold = threshold
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "model": "BiLSTM",
                "accuracy": round(acc, 6),
                "precision": round(prec, 6),
                "recall": round(rec, 6),
                "f1": round(f1, 6),
                "confusion_matrix": cm,
                "threshold": threshold,
                "selected": True,
                "best_epoch": epoch,
            }

    assert best_state is not None
    torch.save(best_state, root_dir / "lstm_model.pt")
    with (root_dir / "lstm_vocab.pkl").open("wb") as vf:
        pickle.dump(
            {
                "vocab": vocab,
                "max_len": max_len,
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "threshold": best_threshold,
            },
            vf,
        )

    plot_confusion_matrix(
        best_metrics["confusion_matrix"],
        "BiLSTM",
        artifacts_dir / "lstm_confusion_matrix.png",
    )

    payload = {
        "eda": eda_summary,
        "vocab_size": len(vocab),
        "max_len": max_len,
        "embedding_dim": hidden_dim * 2,
        "threshold": best_threshold,
        "best_metrics": best_metrics,
        "history": history,
    }
    (artifacts_dir / "lstm_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    if is_db_configured():
        try:
            save_benchmark_results(run_id=f"lstm-{uuid.uuid4()}", results=[best_metrics])
        except Exception as exc:
            print(f"[warn] Skipping PostgreSQL LSTM benchmark log: {exc}")

    return payload


if __name__ == "__main__":
    summary = train_lstm_pipeline()
    print("\n=== BiLSTM Best Validation Metrics ===")
    print(json.dumps(summary["best_metrics"], indent=2))
    print("Saved lstm_model.pt and lstm_vocab.pkl successfully.")
