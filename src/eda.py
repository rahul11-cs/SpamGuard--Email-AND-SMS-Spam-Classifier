"""Phase 1: Dataset loading, cleaning, deduplication, and EDA visualizations."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

from src.preprocess import transform_text

UCI_SMS_SPAM_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
)


def download_sms_spam_dataset(data_dir: Path) -> Path:
    """Download and extract the UCI SMS Spam Collection dataset if missing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target_path = data_dir / "SMSSpamCollection"
    if target_path.exists():
        return target_path

    response = requests.get(UCI_SMS_SPAM_URL, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open("SMSSpamCollection") as src_file:
            target_path.write_bytes(src_file.read())

    return target_path


def load_raw_dataset(dataset_path: Path | None = None, data_dir: Path = Path("data")) -> pd.DataFrame:
    """Load SMS Spam Collection from CSV or UCI tab-separated file."""
    if dataset_path is not None:
        path = Path(dataset_path)
    elif (data_dir / "spam.csv").exists():
        path = data_dir / "spam.csv"
    elif (data_dir / "SMSSpamCollection").exists():
        path = data_dir / "SMSSpamCollection"
    else:
        path = download_sms_spam_dataset(data_dir)

    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

        # Drop junk Unnamed columns common in Kaggle spam.csv
        junk_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
        if junk_cols:
            df = df.drop(columns=junk_cols)

        rename_map = {}
        if "v1" in df.columns and "v2" in df.columns:
            rename_map = {"v1": "label", "v2": "text"}
        elif "Category" in df.columns and "Message" in df.columns:
            rename_map = {"Category": "label", "Message": "text"}
        if rename_map:
            df = df.rename(columns=rename_map)
    else:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["label", "text"],
            encoding="utf-8",
        )

    if not {"label", "text"}.issubset(df.columns):
        raise ValueError(
            f"Expected 'label' and 'text' columns after cleaning, got: {list(df.columns)}"
        )

    return df[["label", "text"]].copy()


def clean_and_prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Clean, deduplicate, encode labels (ham=0, spam=1), and compute EDA metrics."""
    initial_rows = len(df)
    cleaned = df.dropna(subset=["label", "text"]).copy()
    cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()
    cleaned = cleaned[cleaned["label"].isin(["ham", "spam"])].copy()

    duplicates_removed = int(cleaned.duplicated(keep="first").sum())
    cleaned = cleaned.drop_duplicates(keep="first").reset_index(drop=True)

    # Encode ham=0 and spam=1
    cleaned["target"] = cleaned["label"].map({"ham": 0, "spam": 1}).astype(int)

    # Text length & word count features
    cleaned["num_characters"] = cleaned["text"].astype(str).apply(len)
    cleaned["num_words"] = cleaned["text"].astype(str).apply(lambda s: len(s.split()))
    cleaned["transformed_text"] = cleaned["text"].astype(str).apply(transform_text)

    ham_count = int((cleaned["target"] == 0).sum())
    spam_count = int((cleaned["target"] == 1).sum())
    total_clean = len(cleaned)

    summary = {
        "initial_records": initial_rows,
        "duplicates_removed": duplicates_removed,
        "clean_records": total_clean,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "ham_pct": round((ham_count / total_clean) * 100, 2) if total_clean else 0.0,
        "spam_pct": round((spam_count / total_clean) * 100, 2) if total_clean else 0.0,
        "avg_chars_ham": round(float(cleaned.loc[cleaned["target"] == 0, "num_characters"].mean()), 2),
        "avg_chars_spam": round(float(cleaned.loc[cleaned["target"] == 1, "num_characters"].mean()), 2),
        "avg_words_ham": round(float(cleaned.loc[cleaned["target"] == 0, "num_words"].mean()), 2),
        "avg_words_spam": round(float(cleaned.loc[cleaned["target"] == 1, "num_words"].mean()), 2),
    }
    return cleaned, summary


def plot_eda_charts(cleaned: pd.DataFrame, output_dir: Path = Path("artifacts")) -> None:
    """Generate class distribution and top-words-per-class charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Class distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    counts = cleaned["label"].value_counts().reindex(["ham", "spam"])
    palette = ["#2563eb", "#dc2626"]
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=palette, legend=False, ax=axes[0])
    axes[0].set_title("Class Distribution (Count)")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Messages")
    for idx, val in enumerate(counts.values):
        axes[0].text(idx, val + 40, f"{val:,}", ha="center", fontweight="bold")

    axes[1].pie(
        counts.values,
        labels=[f"Ham ({counts['ham'] / len(cleaned) * 100:.1f}%)", f"Spam ({counts['spam'] / len(cleaned) * 100:.1f}%)"],
        colors=palette,
        autopct="%1.2f%%",
        startangle=90,
    )
    axes[1].set_title("Class Imbalance Share")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Top 20 words per class
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, target_val, title, color in [
        (axes[0], 0, "Top 20 Words — Ham (Not Spam)", "#2563eb"),
        (axes[1], 1, "Top 20 Words — Spam", "#dc2626"),
    ]:
        corpus = " ".join(cleaned.loc[cleaned["target"] == target_val, "transformed_text"].tolist()).split()
        top_words = pd.Series(corpus).value_counts().head(20)
        sns.barplot(x=top_words.values, y=top_words.index, color=color, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Stemmed Token")

    fig.tight_layout()
    fig.savefig(output_dir / "top_words.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raw_df = load_raw_dataset()
    clean_df, eda_summary = clean_and_prepare_data(raw_df)
    plot_eda_charts(clean_df)
    print("EDA Summary:", eda_summary)
