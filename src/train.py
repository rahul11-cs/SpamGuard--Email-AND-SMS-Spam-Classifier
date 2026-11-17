"""Phase 3 & 4: TF-IDF feature engineering, 7-classifier benchmarking, and pickle serialization."""

from __future__ import annotations

import json
import pickle
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.db import is_db_configured, save_benchmark_results
from src.eda import clean_and_prepare_data, load_raw_dataset, plot_eda_charts


def build_classifiers() -> Dict[str, Any]:
    """Instantiate the 7 benchmark classifiers."""
    return {
        "MultinomialNB": MultinomialNB(),
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB(),
        "LogisticRegression": LogisticRegression(solver="liblinear", penalty="l1", random_state=2),
        "SVC": SVC(kernel="sigmoid", gamma=1.0, probability=True, random_state=2),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=2, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=50,
            random_state=2,
            eval_metric="logloss",
        ),
    }


def evaluate_models(
    X_train_sparse,
    X_test_sparse,
    y_train,
    y_test,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Train and evaluate all 7 classifiers on the stratified 80/20 split."""
    classifiers = build_classifiers()
    results: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}

    X_train_dense = X_train_sparse.toarray()
    X_test_dense = X_test_sparse.toarray()

    for name, clf in classifiers.items():
        if name == "GaussianNB":
            clf.fit(X_train_dense, y_train)
            y_pred = clf.predict(X_test_dense)
        else:
            clf.fit(X_train_sparse, y_train)
            y_pred = clf.predict(X_test_sparse)

        cm = confusion_matrix(y_test, y_pred)
        results.append(
            {
                "model": name,
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
                "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 6),
                "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 6),
                "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 6),
                "confusion_matrix": cm.tolist(),
                "selected": name == "MultinomialNB",
            }
        )
        trained_models[name] = clf

    results.sort(key=lambda row: (row["precision"], row["accuracy"]), reverse=True)
    return results, trained_models


def plot_confusion_matrix(cm: List[List[int]], model_name: str, output_path: Path) -> None:
    """Save a formatted confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred: Not Spam (0)", "Pred: Spam (1)"],
        yticklabels=["Actual: Not Spam (0)", "Actual: Spam (1)"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name} (TF-IDF 3000)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_training_pipeline(
    dataset_path: Path | None = None,
    artifacts_dir: Path = Path("artifacts"),
    root_dir: Path = Path("."),
    test_size: float = 0.20,
    random_state: int = 2,
    max_features: int = 3000,
) -> Dict[str, Any]:
    """Execute Phases 1–4 end-to-end and serialize vectorizer.pkl and model.pkl."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1 & 2: Load, clean, deduplicate, EDA, and transform_text()
    raw_df = load_raw_dataset(dataset_path=dataset_path)
    clean_df, eda_summary = clean_and_prepare_data(raw_df)
    plot_eda_charts(clean_df, output_dir=artifacts_dir)

    # Phase 3: TF-IDF sparse matrix + stratified 80/20 split
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(clean_df["transformed_text"])
    y = clean_df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    results, trained_models = evaluate_models(X_train, X_test, y_train, y_test)

    # Select MultinomialNB (TF-IDF + MNB prioritizes zero false positives / 100% precision)
    selected_name = "MultinomialNB"
    selected_model = trained_models[selected_name]
    selected_metrics = next(r for r in results if r["model"] == selected_name)

    plot_confusion_matrix(
        selected_metrics["confusion_matrix"],
        selected_name,
        artifacts_dir / "confusion_matrix.png",
    )

    # Phase 4: Serialize vectorizer.pkl and model.pkl
    vectorizer_path = root_dir / "vectorizer.pkl"
    model_path = root_dir / "model.pkl"
    with vectorizer_path.open("wb") as f:
        pickle.dump(tfidf, f)
    with model_path.open("wb") as f:
        pickle.dump(selected_model, f)

    # Save structured EDA & benchmark metadata
    summary_payload = {
        "eda": eda_summary,
        "split": {
            "test_size": test_size,
            "stratified": True,
            "random_state": random_state,
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "max_features": max_features,
        },
        "selected_model": selected_metrics,
        "benchmarks": results,
    }
    (artifacts_dir / "benchmark_results.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(results).to_csv(artifacts_dir / "benchmark_results.csv", index=False)

    if is_db_configured():
        try:
            save_benchmark_results(run_id=str(uuid.uuid4()), results=results)
        except Exception as exc:
            print(f"[warn] Skipping PostgreSQL benchmark log: {exc}")

    return summary_payload


if __name__ == "__main__":
    report = run_training_pipeline()
    print("\n=== EDA Summary ===")
    print(json.dumps(report["eda"], indent=2))
    print("\n=== Model Benchmark Table ===")
    df_table = pd.DataFrame(report["benchmarks"])
    print(df_table[["model", "accuracy", "precision", "recall", "f1", "confusion_matrix"]].to_string(index=False))
    print("\nSelected model:", report["selected_model"])
    print("Saved vectorizer.pkl and model.pkl successfully.")
