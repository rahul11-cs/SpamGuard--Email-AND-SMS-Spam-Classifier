"""FastAPI REST service with OpenAPI docs for MNB, BiLSTM, and PostgreSQL + pgvector."""

from __future__ import annotations

import hmac
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
from src.threat_analyzer import build_model_predictions

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 20_000
MAX_BATCH_SIZE = 100

app = FastAPI(
    title="Email Spam Classification Platform API",
    description="High-throughput FastAPI service running MultinomialNB (TF-IDF) and PyTorch BiLSTM with PostgreSQL + pgvector.",
    version="1.0.0",
)

_allowed_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "https://spam-classifierx.vercel.app",
    ).split(",")
    if origin.strip()
]
if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Admin-Token"],
    )


@app.middleware("http")
async def add_security_headers(request: Request, call_next: Any) -> Any:
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Cache-Control"] = "no-store"
    return response


def _is_admin_authorized(
    authorization: Optional[str] = None,
    x_admin_token: Optional[str] = None,
) -> bool:
    expected = os.getenv("ADMIN_API_TOKEN", "").strip()
    if not expected:
        return False
    provided = ""
    if authorization and authorization.startswith("Bearer "):
        provided = authorization[7:].strip()
    elif x_admin_token:
        provided = x_admin_token.strip()
    return bool(provided) and hmac.compare_digest(provided, expected)


_VECTORIZER = None
_MODEL = None
_LSTM_PREDICTOR: Optional[LstmSpamPredictor] = None


class ClassifyRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_CHARS,
        description="Raw email or SMS text to classify.",
    )
    model: Literal["mnb", "lstm", "both"] = Field(
        default="mnb",
        description="Classifier backend: pre-trained MultinomialNB, BiLSTM, or both.",
    )
    store_in_db: bool = Field(
        default=True,
        description="Persist prediction and embedding in PostgreSQL + pgvector when configured.",
    )


class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="List of messages to classify.",
    )


def _get_mnb_artifacts() -> Tuple[Any, Any]:
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
    global _LSTM_PREDICTOR
    if _LSTM_PREDICTOR is None:
        _LSTM_PREDICTOR = LstmSpamPredictor()
    return _LSTM_PREDICTOR


@app.on_event("startup")
def on_startup() -> None:
    ensure_nltk_data()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "mnb_artifacts_ready": Path("vectorizer.pkl").exists() and Path("model.pkl").exists(),
        "lstm_artifacts_ready": Path("lstm_model.pt").exists() and Path("lstm_vocab.pkl").exists(),
        "postgres_configured": is_db_configured(),
    }


@app.post("/db/init")
def db_init(
    authorization: Optional[str] = Header(default=None),
    x_admin_token: Optional[str] = Header(default=None),
) -> Dict[str, str]:
    if not _is_admin_authorized(authorization=authorization, x_admin_token=x_admin_token):
        raise HTTPException(status_code=403, detail="Forbidden: valid ADMIN_API_TOKEN required.")
    if not is_db_configured():
        raise HTTPException(status_code=400, detail="PostgreSQL environment variables are not configured.")
    init_db()
    return {"status": "initialized"}


@app.post("/predict")
@app.post("/api/v1/classify")
def classify(req: ClassifyRequest) -> Dict[str, Any]:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text must not be empty.")

    try:
        vectorizer, model = _get_mnb_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    t0 = time.perf_counter()
    transformed = transform_text(req.text)
    sparse_vec = vectorizer.transform([transformed])
    pred = int(model.predict(sparse_vec)[0])
    probs = model.predict_proba(sparse_vec)[0]
    spam_prob = float(probs[1])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    label = "Spam" if pred == 1 else "Not Spam"
    dense_tfidf = sparse_vec.toarray()[0]

    lstm_res = None
    lstm_emb = None
    if req.model in {"lstm", "both"}:
        try:
            lstm_res = _get_lstm_predictor().predict(req.text)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        lstm_emb = lstm_res["embedding"]
        if req.model == "lstm":
            pred = int(lstm_res["prediction"])
            label = lstm_res["label"]
            spam_prob = float(lstm_res["spam_probability"])
            elapsed_ms = float(lstm_res["inference_ms"])

    model_used = (
        "MultinomialNB+BiLSTM"
        if req.model == "both"
        else ("BiLSTM" if req.model == "lstm" else "MultinomialNB")
    )

    record_id = None
    similar: List[Dict[str, Any]] = []
    if req.store_in_db and is_db_configured():
        try:
            record_id = log_prediction(
                raw_text=req.text,
                transformed_text=transformed,
                prediction=pred,
                prediction_label=label,
                spam_probability=spam_prob,
                inference_ms=elapsed_ms,
                tfidf_dense_vector=dense_tfidf,
                lstm_embedding=lstm_emb,
                model_used=model_used,
                source="fastapi",
            )
            if lstm_emb is not None and req.model == "lstm":
                similar = find_similar_messages_by_lstm(lstm_emb, limit=5, exclude_id=record_id)
            else:
                similar = find_similar_messages(dense_tfidf, limit=5, exclude_id=record_id)
        except Exception as exc:
            logger.warning("Database logging skipped: %s", exc)
            similar = [{"warning": "Persistence service unavailable."}]

    threat_info = classify_threat_type(req.text, pred)
    payload: Dict[str, Any] = {
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
        "model_predictions": build_model_predictions(
            mnb_spam_prob=float(probs[1]),
            lstm_spam_prob=float(lstm_res["spam_probability"]) if lstm_res is not None else None,
        ),
        "similar_messages": similar,
    }
    if lstm_res is not None and req.model == "both":
        payload["lstm"] = {
            "prediction": lstm_res["prediction"],
            "label": lstm_res["label"],
            "spam_probability": round(float(lstm_res["spam_probability"]), 6),
            "inference_ms": round(float(lstm_res["inference_ms"]), 3),
        }
    return payload


@app.post("/predict/batch")
def classify_batch(req: BatchClassifyRequest) -> Dict[str, Any]:
    vectorizer, model = _get_mnb_artifacts()
    t0 = time.perf_counter()
    sanitized = [str(item)[:MAX_TEXT_CHARS] for item in req.texts]
    transformed_list = [transform_text(item) for item in sanitized]
    sparse_matrix = vectorizer.transform(transformed_list)
    preds = model.predict(sparse_matrix)
    probs = model.predict_proba(sparse_matrix)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "count": len(sanitized),
        "inference_ms": round(elapsed_ms, 3),
        "results": [
            {
                "text": raw,
                "prediction": int(p),
                "label": "Spam" if int(p) == 1 else "Not Spam",
                **classify_threat_type(raw, int(p)),
                "spam_probability": round(float(pr[1]), 6),
            }
            for raw, p, pr in zip(sanitized, preds, probs)
        ],
    }


@app.get("/history")
def history(
    limit: int = Query(default=20, ge=1, le=100),
    authorization: Optional[str] = Header(default=None),
    x_admin_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    public_enabled = os.getenv("ENABLE_PUBLIC_HISTORY", "false").strip().lower() == "true"
    if not public_enabled and not _is_admin_authorized(authorization=authorization, x_admin_token=x_admin_token):
        raise HTTPException(status_code=403, detail="History endpoint is disabled in public mode.")
    if not is_db_configured():
        return {"results": [], "note": "PostgreSQL not configured."}
    return {"results": get_recent_predictions(limit=limit)}


@app.get("/benchmarks")
def benchmarks() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    bench_file = Path("artifacts/benchmark_results.json")
    lstm_file = Path("artifacts/lstm_metrics.json")
    if bench_file.exists():
        out["sklearn_benchmarks"] = json.loads(bench_file.read_text(encoding="utf-8"))
    if lstm_file.exists():
        out["lstm_benchmark"] = json.loads(lstm_file.read_text(encoding="utf-8"))
    if not out:
        raise HTTPException(status_code=404, detail="Benchmark artifacts not generated yet.")
    return out
