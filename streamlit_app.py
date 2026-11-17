"""Phase 5 & 6: Streamlit web application for real-time Email/SMS Spam Classification."""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

from src.db import (
    find_similar_messages,
    find_similar_messages_by_lstm,
    get_recent_predictions,
    is_db_configured,
    log_prediction,
)
from src.lstm_model import LstmSpamPredictor
from src.preprocess import classify_threat_type, ensure_nltk_data, transform_text
from src.threat_analyzer import BENCHMARK_DATA, build_model_predictions
from src.train import run_training_pipeline

st.set_page_config(
    page_title="Email Spam Classification Platform",
    page_icon="🛡️",
    layout="wide",
)

# Phase 6: Download required NLTK corpora at app startup (for Streamlit Community Cloud)
ensure_nltk_data()


@st.cache_resource
def load_model_artifacts() -> Tuple[Any, Any]:
    """Load TF-IDF vectorizer and Multinomial Naive Bayes model once per server process."""
    vec_path = Path("vectorizer.pkl")
    model_path = Path("model.pkl")

    if not vec_path.exists() or not model_path.exists():
        run_training_pipeline()

    with vec_path.open("rb") as vf:
        vectorizer = pickle.load(vf)
    with model_path.open("rb") as mf:
        model = pickle.load(mf)
    return vectorizer, model


@st.cache_resource
def load_lstm_predictor() -> Optional[LstmSpamPredictor]:
    """Load PyTorch BiLSTM artifacts if trained."""
    if not Path("lstm_model.pt").exists() or not Path("lstm_vocab.pkl").exists():
        return None
    return LstmSpamPredictor()


def load_benchmark_report() -> Dict[str, Any] | None:
    """Load saved Phase 1–3 EDA and benchmark JSON if present."""
    bench_path = Path("artifacts/benchmark_results.json")
    if not bench_path.exists():
        return None
    return json.loads(bench_path.read_text(encoding="utf-8"))


def load_lstm_report() -> Dict[str, Any] | None:
    """Load saved BiLSTM evaluation metrics if present."""
    lstm_path = Path("artifacts/lstm_metrics.json")
    if not lstm_path.exists():
        return None
    return json.loads(lstm_path.read_text(encoding="utf-8"))


def main() -> None:
    st.title("Email & SMS Spam Classification Platform")
    st.caption(
        "NLP Pipeline (Regex + NLTK Stopwords + Porter Stemming + TF-IDF 3000) | "
        "Multinomial Naive Bayes & PyTorch BiLSTM | Flask & FastAPI | PostgreSQL + pgvector"
    )

    vectorizer, model = load_model_artifacts()
    lstm_predictor = load_lstm_predictor()
    default_api_url = os.getenv("FLASK_API_URL", "").strip().rstrip("/")
    db_enabled = is_db_configured()

    with st.sidebar:
        st.header("Runtime Configuration")
        inference_mode = st.radio(
            "Inference Backend",
            options=[
                "Local Cached MNB (@st.cache_resource)",
                "Local BiLSTM (PyTorch)",
                "Flask / FastAPI REST Endpoint",
            ],
            index=0,
        )
        api_base_url = default_api_url or "http://localhost:5000"
        api_model_param = "mnb"
        if inference_mode == "Flask / FastAPI REST Endpoint":
            api_base_url = st.text_input(
                "REST API Base URL",
                value=api_base_url,
            ).strip().rstrip("/")
            api_model_param = st.selectbox("API Model", ["mnb", "lstm", "both"], index=0)

    tab_classify, tab_benchmarks, tab_eda, tab_history = st.tabs(
        ["Real-Time Classifier", "Model Benchmarks & LSTM", "EDA & Top Words", "PostgreSQL + pgvector History"]
    )

    with tab_classify:
        st.subheader("Classify Email or SMS Content")
        col_s, col_sub = st.columns(2)
        sender_text = col_s.text_input("Sender Address", value="", max_chars=320, placeholder="sender@domain.com")
        subject_text = col_sub.text_input("Subject Line", value="", max_chars=500, placeholder="Email subject...")
        user_text = st.text_area(
            "Message Body",
            value="",
            max_chars=20000,
            height=180,
            placeholder="Paste an email body or SMS message here...",
        )

        predict_clicked = st.button("Predict", type="primary", use_container_width=False)

        if predict_clicked:
            combined_text = " ".join(
                part.strip() for part in (subject_text, user_text) if part and part.strip()
            )
            if not combined_text:
                st.warning("Please enter a Subject Line or Message Body before clicking Predict.")
            else:
                similar_rows = []
                if inference_mode == "Flask / FastAPI REST Endpoint":
                    if not api_base_url.startswith(("http://", "https://")):
                        st.error("REST API Base URL must start with http:// or https://.")
                        st.stop()
                    t0 = time.perf_counter()
                    try:
                        resp = requests.post(
                            f"{api_base_url}/predict",
                            json={"text": combined_text, "model": api_model_param},
                            timeout=10,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        roundtrip_ms = (time.perf_counter() - t0) * 1000.0
                        label = data["label"]
                        spam_prob = float(data["spam_probability"])
                        transformed = data["transformed_text"]
                        inference_ms = float(data["inference_ms"])
                        threat_info = {
                            "threat_type": data.get("threat_type", "Unknown"),
                            "threat_risk_level": data.get("threat_risk_level", "None"),
                            "threat_reason": data.get("threat_reason", ""),
                        }
                        similar_rows = data.get("similar_messages", [])
                        st.caption(f"REST API round-trip: {roundtrip_ms:.2f} ms")
                    except Exception:
                        st.error("REST API request failed. Verify the backend URL and try again.")
                        st.stop()
                elif inference_mode == "Local BiLSTM (PyTorch)":
                    if lstm_predictor is None:
                        st.warning("BiLSTM model not trained yet. Run `python -m src.train_lstm` first.")
                        st.stop()
                    res = lstm_predictor.predict(combined_text)
                    label = res["label"]
                    spam_prob = float(res["spam_probability"])
                    transformed = res["transformed_text"]
                    inference_ms = float(res["inference_ms"])
                    threat_info = classify_threat_type(
                        f"{sender_text} {combined_text}", int(res["prediction"])
                    )

                    if db_enabled:
                        try:
                            inserted_id = log_prediction(
                                raw_text=combined_text,
                                transformed_text=transformed,
                                prediction=res["prediction"],
                                prediction_label=label,
                                spam_probability=spam_prob,
                                inference_ms=inference_ms,
                                lstm_embedding=res["embedding"],
                                model_used="BiLSTM",
                                source="streamlit-lstm",
                            )
                            similar_rows = find_similar_messages_by_lstm(
                                lstm_embedding=res["embedding"],
                                limit=5,
                                exclude_id=inserted_id,
                            )
                        except Exception:
                            st.info("PostgreSQL/pgvector persistence unavailable.")
                else:
                    t0 = time.perf_counter()
                    transformed = transform_text(combined_text)
                    sparse_vec = vectorizer.transform([transformed])
                    prediction = int(model.predict(sparse_vec)[0])
                    probs = model.predict_proba(sparse_vec)[0]
                    spam_prob = float(probs[1])
                    inference_ms = (time.perf_counter() - t0) * 1000.0
                    label = "Spam" if prediction == 1 else "Not Spam"
                    threat_info = classify_threat_type(
                        f"{sender_text} {combined_text}", prediction
                    )

                    if db_enabled:
                        try:
                            dense_vec = sparse_vec.toarray()[0]
                            inserted_id = log_prediction(
                                raw_text=combined_text,
                                transformed_text=transformed,
                                prediction=prediction,
                                prediction_label=label,
                                spam_probability=spam_prob,
                                inference_ms=inference_ms,
                                tfidf_dense_vector=dense_vec,
                                model_used="MultinomialNB",
                                source="streamlit",
                            )
                            similar_rows = find_similar_messages(
                                tfidf_dense_vector=dense_vec,
                                limit=5,
                                exclude_id=inserted_id,
                            )
                        except Exception:
                            st.info("PostgreSQL/pgvector persistence unavailable.")

                if label == "Spam":
                    st.error(f"Result: **Spam** (Spam Probability: {spam_prob * 100:.2f}%)")
                else:
                    st.success(f"Result: **Not Spam** (Ham Confidence: {(1.0 - spam_prob) * 100:.2f}%)")

                st.info(
                    f"**Threat / Scam Type:** {threat_info['threat_type']} "
                    f"(Risk: {threat_info['threat_risk_level']}) — {threat_info['threat_reason']}"
                )

                col1, col2, col3 = st.columns(3)
                col1.metric("Prediction", label)
                col2.metric("Inference Time", f"{inference_ms:.3f} ms ({inference_ms / 1000.0:.5f} s)")
                col3.metric("Spam Probability", f"{spam_prob * 100:.2f}%")

                # Per-model prediction accuracy breakdown on provided mail
                mnb_sparse_vec = vectorizer.transform([transformed])
                mnb_prob_val = float(model.predict_proba(mnb_sparse_vec)[0][1])
                lstm_prob_val = (
                    float(lstm_predictor.predict(combined_text)["spam_probability"])
                    if lstm_predictor is not None
                    else (spam_prob if inference_mode == "Local BiLSTM (PyTorch)" else None)
                )
                per_model_rows = build_model_predictions(
                    mnb_spam_prob=mnb_prob_val,
                    lstm_spam_prob=lstm_prob_val,
                )
                st.markdown("#### Per-Model Prediction Accuracy on Provided Mail")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Model": item["model"],
                                "Verdict": item["label"],
                                "Prediction Confidence": f"{item['confidence'] * 100:.2f}%",
                                "P(Spam)": f"{item['spam_probability'] * 100:.2f}%",
                                "Validation Accuracy": f"{item['validation_accuracy'] * 100:.2f}%",
                            }
                            for item in per_model_rows
                        ]
                    ),
                    use_container_width=True,
                )

                with st.expander("Preprocessed Tokens (transform_text output)"):
                    st.code(transformed or "(empty after stopword/regex filtering)")

                if similar_rows:
                    st.markdown("#### Similar Historical Messages (pgvector Cosine Similarity)")
                    st.dataframe(pd.DataFrame(similar_rows), use_container_width=True)

    with tab_benchmarks:
        st.subheader("7-Classifier Benchmark Comparison & BiLSTM Evaluation")
        report = load_benchmark_report()
        lstm_report = load_lstm_report()

        if report is None:
            st.info("Run `python -m src.train` to generate benchmark tables and confusion matrix plots.")
        else:
            rows = list(report["benchmarks"])
            if lstm_report and "best_metrics" in lstm_report:
                rows.append(lstm_report["best_metrics"])
            rows.sort(
                key=lambda r: (float(r.get("precision", 0)), float(r.get("accuracy", 0)), float(r.get("f1", 0))),
                reverse=True,
            )
            bench_df = pd.DataFrame(rows)
            st.dataframe(
                bench_df[["model", "accuracy", "precision", "recall", "f1", "confusion_matrix", "selected"]],
                use_container_width=True,
            )
            c_left, c_right = st.columns(2)
            bilstm_summary = next(
                (r for r in rows if str(r.get("model", "")).startswith("BiLSTM")),
                BENCHMARK_DATA[0],
            )
            mnb_summary = next(
                (r for r in rows if r.get("model") == "MultinomialNB"),
                BENCHMARK_DATA[1],
            )
            for col_box, title, item, img_path in [
                (c_left, "BiLSTM (PyTorch 128d)", bilstm_summary, Path("artifacts/lstm_confusion_matrix.png")),
                (c_right, "MultinomialNB", mnb_summary, Path("artifacts/confusion_matrix.png")),
            ]:
                with col_box:
                    st.markdown(f"**Confusion Matrix — {title}**")
                    cm_vals = item.get("confusion_matrix", [[0, 0], [0, 0]])
                    q1, q2 = st.columns(2)
                    q1.metric("True Negative (Ham)", int(cm_vals[0][0]))
                    q2.metric("False Positive", int(cm_vals[0][1]))
                    q3, q4 = st.columns(2)
                    q3.metric("False Negative", int(cm_vals[1][0]))
                    q4.metric("True Positive (Spam)", int(cm_vals[1][1]))
                    if img_path.exists():
                        st.image(str(img_path), caption=f"{title} Confusion Matrix")

    with tab_eda:
        st.subheader("Exploratory Data Analysis (SMS Spam Collection)")
        report = load_benchmark_report()
        if report and "eda" in report:
            eda = report["eda"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Raw Records", f"{eda['initial_records']:,}")
            c2.metric("Duplicates Removed", f"{eda['duplicates_removed']:,}")
            c3.metric("Ham Share", f"{eda['ham_pct']}%")
            c4.metric("Spam Share", f"{eda['spam_pct']}%")

        dist_img = Path("artifacts/class_distribution.png")
        words_img = Path("artifacts/top_words.png")
        if dist_img.exists():
            st.image(str(dist_img), caption="Class Imbalance Distribution", use_container_width=True)
        if words_img.exists():
            st.image(str(words_img), caption="Top 20 Stemmed Tokens by Class", use_container_width=True)

    with tab_history:
        st.subheader("PostgreSQL + pgvector Classification Log")
        public_history = os.getenv("ENABLE_PUBLIC_HISTORY", "false").strip().lower() == "true"
        if not public_history:
            st.info(
                "Shared database history is disabled in public mode (`ENABLE_PUBLIC_HISTORY=false`) "
                "to protect submitter message privacy."
            )
        elif not db_enabled:
            st.info(
                "Configure `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, and `PGPASSWORD` in environment secrets "
                "and run `psql -f db/schema.sql` to enable persistent classification logs."
            )
        else:
            try:
                recent = get_recent_predictions(limit=25)
                if recent:
                    st.dataframe(pd.DataFrame(recent), use_container_width=True)
                else:
                    st.info("No predictions logged yet.")
            except Exception:
                st.warning("PostgreSQL logs are currently unavailable.")

    # Bottom 4-Metric Summary Strip
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BiLSTM Accuracy", "98.55%")
    m2.metric("BiLSTM Precision", "100.0%")
    m3.metric("Training Corpus", "5,572")
    m4.metric("TF-IDF Features", "3,000")


if __name__ == "__main__":
    main()
