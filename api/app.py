"""Flask REST API for real-time MNB & BiLSTM spam classification and PostgreSQL + pgvector storage."""

from __future__ import annotations

import hmac
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, render_template, request

from src.db import (
    find_similar_messages,
    find_similar_messages_by_lstm,
    get_recent_predictions,
    init_db,
    is_db_configured,
    log_prediction,
)
from src.lstm_model import LstmSpamPredictor
from src.preprocess import classify_threat_type, ensure_nltk_data, transform_text
from src.threat_analyzer import BENCHMARK_DATA, analyze_email_payload, build_model_predictions

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 20_000
MAX_BATCH_SIZE = 100

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB request cap


def _is_admin_authorized() -> bool:
    """Validate optional ADMIN_API_TOKEN via constant-time comparison (fail-closed)."""
    expected = os.getenv("ADMIN_API_TOKEN", "").strip()
    if not expected:
        return False
    auth_header = request.headers.get("Authorization", "")
    provided = auth_header[7:].strip() if auth_header.startswith("Bearer ") else request.headers.get("X-Admin-Token", "").strip()
    return bool(provided) and hmac.compare_digest(provided, expected)


def _public_history_enabled() -> bool:
    return os.getenv("ENABLE_PUBLIC_HISTORY", "false").strip().lower() == "true"


@app.after_request
def _apply_security_headers(response: Any) -> Any:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    if request.path.startswith(("/predict", "/api/", "/history", "/health", "/db/")):
        response.headers["Cache-Control"] = "no-store"

    allowed_origins = {
        origin.strip().rstrip("/")
        for origin in os.getenv(
            "ALLOWED_ORIGINS",
            "https://spam-classifierx.vercel.app",
        ).split(",")
        if origin.strip()
    }
    origin = request.headers.get("Origin", "")
    if origin and origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Admin-Token"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

_VECTORIZER = None
_MODEL = None
_LSTM_PREDICTOR: Optional[LstmSpamPredictor] = None


def _get_mnb_artifacts() -> Tuple[Any, Any]:
    """Load serialized TF-IDF vectorizer and MultinomialNB model once."""
    global _VECTORIZER, _MODEL
    if _VECTORIZER is None or _MODEL is None:
        vec_path = Path("vectorizer.pkl")
        model_path = Path("model.pkl")
        if not vec_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                "Missing vectorizer.pkl or model.pkl. Run `python -m src.train` first."
            )
        with vec_path.open("rb") as vf:
            _VECTORIZER = pickle.load(vf)
        with model_path.open("rb") as mf:
            _MODEL = pickle.load(mf)
    return _VECTORIZER, _MODEL


def _get_lstm_predictor() -> LstmSpamPredictor:
    """Load serialized PyTorch BiLSTM artifacts once."""
    global _LSTM_PREDICTOR
    if _LSTM_PREDICTOR is None:
        _LSTM_PREDICTOR = LstmSpamPredictor()
    return _LSTM_PREDICTOR


@app.before_request
def _bootstrap_nltk() -> None:
    ensure_nltk_data()


@app.route("/", methods=["GET", "POST"])
def home() -> Any:
    """Serve pure Python Flask HTML UI with empty fields, threat-type detection, and bottom KPI summary."""
    form_sender = ""
    form_subject = ""
    form_body = ""
    result = None
    error = None

    if request.method == "POST":
        form_sender = request.form.get("sender", "")
        form_subject = request.form.get("subject", "")
        form_body = request.form.get("body", "")
        if not form_subject.strip() and not form_body.strip():
            error = "Enter a Subject Line or Message Body before classifying."
        else:
            vectorizer, model = _get_mnb_artifacts()
            lstm_pred = (
                _get_lstm_predictor()
                if Path("lstm_model.pt").exists() and Path("lstm_vocab.pkl").exists()
                else None
            )
            result = analyze_email_payload(
                vectorizer=vectorizer,
                model=model,
                body=form_body,
                subject=form_subject,
                sender=form_sender,
                lstm_predictor=lstm_pred,
            )

    bilstm_cm = next((r for r in BENCHMARK_DATA if r["model"].startswith("BiLSTM")), BENCHMARK_DATA[0])
    mnb_cm = next((r for r in BENCHMARK_DATA if r["model"] == "MultinomialNB"), BENCHMARK_DATA[1])

    return render_template(
        "index.html",
        form_sender=form_sender,
        form_subject=form_subject,
        form_body=form_body,
        result=result,
        error=error,
        benchmarks=BENCHMARK_DATA,
        bilstm_cm=bilstm_cm,
        mnb_cm=mnb_cm,
    )


@app.get("/health")
def health() -> Any:
    """Service health check endpoint."""
    mnb_ready = Path("vectorizer.pkl").exists() and Path("model.pkl").exists()
    lstm_ready = Path("lstm_model.pt").exists() and Path("lstm_vocab.pkl").exists()
    return jsonify(
        {
            "status": "ok",
            "mnb_artifacts_ready": mnb_ready,
            "lstm_artifacts_ready": lstm_ready,
            "postgres_configured": is_db_configured(),
        }
    )


@app.post("/db/init")
def initialize_database() -> Any:
    """Apply PostgreSQL + pgvector schema (requires ADMIN_API_TOKEN)."""
    if not _is_admin_authorized():
        return jsonify({"error": "Forbidden: valid ADMIN_API_TOKEN required."}), 403
    if not is_db_configured():
        return jsonify({"error": "PostgreSQL environment variables are not configured."}), 400
    init_db()
    return jsonify({"status": "initialized"})


@app.post("/predict")
@app.post("/api/v1/classify")
def predict() -> Any:
    """Receive new email text, run pre-trained MultinomialNB/BiLSTM pipeline, and store results in PostgreSQL."""
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    model_choice = str(payload.get("model", "mnb")).lower().strip()

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Request JSON must include non-empty 'text' string."}), 400
    if len(text) > MAX_TEXT_CHARS:
        return jsonify({"error": f"'text' exceeds maximum length of {MAX_TEXT_CHARS} characters."}), 400
    if model_choice not in {"mnb", "lstm", "both"}:
        return jsonify({"error": "'model' must be one of: 'mnb', 'lstm', 'both'."}), 400

    vectorizer, model = _get_mnb_artifacts()

    t0 = time.perf_counter()
    transformed = transform_text(text)
    sparse_vec = vectorizer.transform([transformed])
    pred = int(model.predict(sparse_vec)[0])
    probs = model.predict_proba(sparse_vec)[0]
    spam_prob = float(probs[1])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    label = "Spam" if pred == 1 else "Not Spam"

    dense_tfidf = sparse_vec.toarray()[0]
    lstm_result = None
    lstm_embedding = None

    if model_choice in {"lstm", "both"}:
        predictor = _get_lstm_predictor()
        lstm_result = predictor.predict(text)
        lstm_embedding = lstm_result["embedding"]
        if model_choice == "lstm":
            pred = int(lstm_result["prediction"])
            label = lstm_result["label"]
            spam_prob = float(lstm_result["spam_probability"])
            elapsed_ms = float(lstm_result["inference_ms"])

    record_id = None
    similar_messages = []
    model_used = (
        "MultinomialNB+BiLSTM"
        if model_choice == "both"
        else ("BiLSTM" if model_choice == "lstm" else "MultinomialNB")
    )

    if is_db_configured():
        try:
            record_id = log_prediction(
                raw_text=text,
                transformed_text=transformed,
                prediction=pred,
                prediction_label=label,
                spam_probability=spam_prob,
                inference_ms=elapsed_ms,
                tfidf_dense_vector=dense_tfidf,
                lstm_embedding=lstm_embedding,
                model_used=model_used,
                source="flask",
            )
            if lstm_embedding is not None and model_choice == "lstm":
                similar_messages = find_similar_messages_by_lstm(
                    lstm_embedding=lstm_embedding,
                    limit=5,
                    exclude_id=record_id,
                )
            else:
                similar_messages = find_similar_messages(
                    tfidf_dense_vector=dense_tfidf,
                    limit=5,
                    exclude_id=record_id,
                )
        except Exception as exc:
            logger.warning("Database logging skipped: %s", exc)
            similar_messages = [{"warning": "Persistence service unavailable."}]

    threat_info = classify_threat_type(text, pred)
    mnb_spam_prob = float(probs[1])
    lstm_spam_prob = float(lstm_result["spam_probability"]) if lstm_result is not None else None
    response: Dict[str, Any] = {
        "id": record_id,
        "model_used": model_used,
        "prediction": pred,
        "label": label,
        "is_spam": pred == 1,
        "threat_type": threat_info["threat_type"],
        "threat_risk_level": threat_info["threat_risk_level"],
        "threat_reason": threat_info["threat_reason"],
        "spam_probability": round(spam_prob, 6),
        "ham_probability": round(1.0 - spam_prob, 6),
        "inference_ms": round(elapsed_ms, 3),
        "transformed_text": transformed,
        "embedding_stored": bool(record_id is not None),
        "tfidf_embedding_dim": int(dense_tfidf.shape[0]),
        "model_predictions": build_model_predictions(
            mnb_spam_prob=mnb_spam_prob,
            lstm_spam_prob=lstm_spam_prob,
        ),
        "similar_messages": similar_messages,
    }
    if lstm_result is not None and model_choice == "both":
        response["lstm"] = {
            "prediction": lstm_result["prediction"],
            "label": lstm_result["label"],
            "spam_probability": round(float(lstm_result["spam_probability"]), 6),
            "inference_ms": round(float(lstm_result["inference_ms"]), 3),
            "embedding_dim": int(lstm_result["embedding"].shape[0]),
        }
    return jsonify(response)


@app.post("/predict/lstm")
def predict_lstm() -> Any:
    """Classify email text with the trained BiLSTM and store 128-dim embedding in pgvector."""
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Request JSON must include non-empty 'text' string."}), 400
    if len(text) > MAX_TEXT_CHARS:
        return jsonify({"error": f"'text' exceeds maximum length of {MAX_TEXT_CHARS} characters."}), 400

    predictor = _get_lstm_predictor()
    res = predictor.predict(text)
    record_id = None
    similar_messages = []

    if is_db_configured():
        try:
            record_id = log_prediction(
                raw_text=text,
                transformed_text=res["transformed_text"],
                prediction=res["prediction"],
                prediction_label=res["label"],
                spam_probability=res["spam_probability"],
                inference_ms=res["inference_ms"],
                tfidf_dense_vector=None,
                lstm_embedding=res["embedding"],
                model_used="BiLSTM",
                source="flask-lstm",
            )
            similar_messages = find_similar_messages_by_lstm(
                lstm_embedding=res["embedding"],
                limit=5,
                exclude_id=record_id,
            )
        except Exception as exc:
            logger.warning("Database logging skipped: %s", exc)
            similar_messages = [{"warning": "Persistence service unavailable."}]

    threat_info = classify_threat_type(text, int(res["prediction"]))
    return jsonify(
        {
            "id": record_id,
            "model_used": "BiLSTM",
            "prediction": res["prediction"],
            "label": res["label"],
            "is_spam": res["prediction"] == 1,
            "threat_type": threat_info["threat_type"],
            "threat_risk_level": threat_info["threat_risk_level"],
            "threat_reason": threat_info["threat_reason"],
            "spam_probability": round(float(res["spam_probability"]), 6),
            "ham_probability": round(float(res["ham_probability"]), 6),
            "inference_ms": round(float(res["inference_ms"]), 3),
            "transformed_text": res["transformed_text"],
            "lstm_embedding_dim": int(res["embedding"].shape[0]),
            "similar_messages": similar_messages,
        }
    )


@app.post("/predict/batch")
def predict_batch() -> Any:
    """Classify a batch of messages in a single vectorized MultinomialNB pass."""
    payload = request.get_json(silent=True) or {}
    texts = payload.get("texts")
    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "Request JSON must include non-empty 'texts' list."}), 400
    if len(texts) > MAX_BATCH_SIZE:
        return jsonify({"error": f"'texts' batch exceeds limit of {MAX_BATCH_SIZE} items."}), 400
    sanitized_texts = [str(item)[:MAX_TEXT_CHARS] for item in texts]

    vectorizer, model = _get_mnb_artifacts()

    t0 = time.perf_counter()
    transformed_list = [transform_text(item) for item in sanitized_texts]
    sparse_matrix = vectorizer.transform(transformed_list)
    preds = model.predict(sparse_matrix)
    probs = model.predict_proba(sparse_matrix)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    items = [
        {
            "text": raw,
            "prediction": int(pred),
            "label": "Spam" if int(pred) == 1 else "Not Spam",
            **classify_threat_type(raw, int(pred)),
            "spam_probability": round(float(prob[1]), 6),
        }
        for raw, pred, prob in zip(sanitized_texts, preds, probs)
    ]
    return jsonify({"count": len(items), "inference_ms": round(elapsed_ms, 3), "results": items})


@app.get("/history")
def history() -> Any:
    """Return recent predictions only when public history is enabled or admin token is valid."""
    if not _public_history_enabled() and not _is_admin_authorized():
        return jsonify({"error": "History endpoint is disabled in public mode."}), 403
    if not is_db_configured():
        return jsonify({"results": [], "note": "PostgreSQL not configured."})
    try:
        raw_limit = int(request.args.get("limit", 20))
    except (TypeError, ValueError):
        return jsonify({"error": "'limit' must be an integer between 1 and 100."}), 400
    limit = min(max(raw_limit, 1), 100)
    return jsonify({"results": get_recent_predictions(limit=limit)})


@app.get("/benchmarks")
def benchmarks() -> Any:
    """Return saved Phase 3 MNB & BiLSTM benchmark tables."""
    response: Dict[str, Any] = {}
    bench_file = Path("artifacts/benchmark_results.json")
    lstm_file = Path("artifacts/lstm_metrics.json")
    if bench_file.exists():
        response["sklearn_benchmarks"] = json.loads(bench_file.read_text(encoding="utf-8"))
    if lstm_file.exists():
        response["lstm_benchmark"] = json.loads(lstm_file.read_text(encoding="utf-8"))
    if not response:
        return jsonify({"error": "Benchmark results not found. Run `python -m src.train` first."}), 404
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
