"""PostgreSQL + pgvector persistence layer for classification logs, embeddings, and benchmarks."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from dotenv import load_dotenv
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from pgvector.psycopg2 import register_vector

load_dotenv()

_POOL: Optional[ThreadedConnectionPool] = None


def is_db_configured() -> bool:
    """Return True if explicit PostgreSQL credentials or DATABASE_URL are configured."""
    if os.getenv("DATABASE_URL"):
        return True
    return bool(
        os.getenv("PGHOST")
        and os.getenv("PGDATABASE")
        and os.getenv("PGUSER")
        and os.getenv("PGPASSWORD")
    )


def _get_pool() -> ThreadedConnectionPool:
    """Lazily initialize a thread-safe PostgreSQL connection pool without hardcoded credentials."""
    global _POOL
    if _POOL is None:
        if not is_db_configured():
            raise RuntimeError("PostgreSQL connection environment variables are not configured.")
        database_url = os.getenv("DATABASE_URL", "").strip()
        sslmode = os.getenv("PGSSLMODE", "prefer").strip() or "prefer"
        if database_url:
            _POOL = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=database_url,
                sslmode=sslmode,
                connect_timeout=5,
            )
        else:
            _POOL = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=os.environ["PGHOST"],
                port=int(os.getenv("PGPORT", "5432")),
                dbname=os.environ["PGDATABASE"],
                user=os.environ["PGUSER"],
                password=os.environ["PGPASSWORD"],
                sslmode=sslmode,
                connect_timeout=5,
            )
    return _POOL


@contextmanager
def get_connection() -> Iterator[psycopg2.extensions.connection]:
    """Borrow a pooled connection with pgvector registered."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def init_db(schema_path: Path = Path("db/schema.sql")) -> None:
    """Apply PostgreSQL + pgvector schema."""
    sql = schema_path.read_text(encoding="utf-8")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


def run_migrations(migrations_dir: Path = Path("db/migrations")) -> List[str]:
    """Execute idempotent forward SQL migration scripts in lexical order."""
    if not is_db_configured() or not migrations_dir.exists():
        return []

    migration_files = sorted(
        path
        for path in migrations_dir.glob("*.sql")
        if not path.name.endswith(".down.sql")
    )
    applied: List[str] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            for sql_file in migration_files:
                cur.execute(sql_file.read_text(encoding="utf-8"))
                applied.append(sql_file.name)
    return applied


def log_prediction(
    raw_text: str,
    transformed_text: str,
    prediction: int,
    prediction_label: str,
    spam_probability: float,
    inference_ms: float,
    tfidf_dense_vector: Optional[np.ndarray] = None,
    lstm_embedding: Optional[np.ndarray] = None,
    model_used: str = "MultinomialNB",
    source: str = "streamlit",
) -> Optional[int]:
    """Persist a classified message and its TF-IDF and/or LSTM embeddings in PostgreSQL + pgvector."""
    if not is_db_configured():
        return None

    tfidf_list = (
        np.asarray(tfidf_dense_vector, dtype=np.float32).ravel().tolist()
        if tfidf_dense_vector is not None
        else None
    )
    lstm_list = (
        np.asarray(lstm_embedding, dtype=np.float32).ravel().tolist()
        if lstm_embedding is not None
        else None
    )

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO classification_logs (
                    raw_text,
                    transformed_text,
                    prediction,
                    prediction_label,
                    spam_probability,
                    inference_ms,
                    model_used,
                    tfidf_vector,
                    lstm_embedding,
                    source
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    raw_text,
                    transformed_text,
                    int(prediction),
                    prediction_label,
                    float(spam_probability),
                    float(inference_ms),
                    model_used,
                    tfidf_list,
                    lstm_list,
                    source,
                ),
            )
            row = cur.fetchone()
            return int(row[0]) if row else None


def find_similar_messages(
    tfidf_dense_vector: np.ndarray,
    limit: int = 5,
    exclude_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Query top-k nearest historical messages using pgvector cosine distance (<=>) on TF-IDF."""
    if not is_db_configured():
        return []

    vector_list = np.asarray(tfidf_dense_vector, dtype=np.float32).ravel().tolist()
    with get_connection() as conn:
        with conn.cursor() as cur:
            if exclude_id is not None:
                cur.execute(
                    """
                    SELECT
                        id,
                        raw_text,
                        prediction_label,
                        spam_probability,
                        1 - (tfidf_vector <=> %s::vector) AS cosine_similarity,
                        created_at
                    FROM classification_logs
                    WHERE tfidf_vector IS NOT NULL AND id <> %s
                    ORDER BY tfidf_vector <=> %s::vector
                    LIMIT %s;
                    """,
                    (vector_list, exclude_id, vector_list, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        id,
                        raw_text,
                        prediction_label,
                        spam_probability,
                        1 - (tfidf_vector <=> %s::vector) AS cosine_similarity,
                        created_at
                    FROM classification_logs
                    WHERE tfidf_vector IS NOT NULL
                    ORDER BY tfidf_vector <=> %s::vector
                    LIMIT %s;
                    """,
                    (vector_list, vector_list, limit),
                )
            rows = cur.fetchall()

    return [
        {
            "id": int(r[0]),
            "prediction_label": r[2],
            "spam_probability": float(r[3]),
            "cosine_similarity": round(float(r[4]), 4),
            "created_at": r[5].isoformat() if r[5] else None,
        }
        for r in rows
    ]


def find_similar_messages_by_lstm(
    lstm_embedding: np.ndarray,
    limit: int = 5,
    exclude_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Query top-k nearest historical messages using HNSW-indexed 128-dim LSTM embeddings."""
    if not is_db_configured():
        return []

    vector_list = np.asarray(lstm_embedding, dtype=np.float32).ravel().tolist()
    with get_connection() as conn:
        with conn.cursor() as cur:
            if exclude_id is not None:
                cur.execute(
                    """
                    SELECT
                        id,
                        raw_text,
                        prediction_label,
                        spam_probability,
                        1 - (lstm_embedding <=> %s::vector) AS cosine_similarity,
                        created_at
                    FROM classification_logs
                    WHERE lstm_embedding IS NOT NULL AND id <> %s
                    ORDER BY lstm_embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (vector_list, exclude_id, vector_list, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        id,
                        raw_text,
                        prediction_label,
                        spam_probability,
                        1 - (lstm_embedding <=> %s::vector) AS cosine_similarity,
                        created_at
                    FROM classification_logs
                    WHERE lstm_embedding IS NOT NULL
                    ORDER BY lstm_embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (vector_list, vector_list, limit),
                )
            rows = cur.fetchall()

    return [
        {
            "id": int(r[0]),
            "prediction_label": r[2],
            "spam_probability": float(r[3]),
            "cosine_similarity": round(float(r[4]), 4),
            "created_at": r[5].isoformat() if r[5] else None,
        }
        for r in rows
    ]


def get_recent_predictions(limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch recent classification logs from PostgreSQL."""
    if not is_db_configured():
        return []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    raw_text,
                    prediction_label,
                    spam_probability,
                    inference_ms,
                    model_used,
                    source,
                    created_at
                FROM classification_logs
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return [
        {
            "id": int(r[0]),
            "raw_text": r[1],
            "prediction_label": r[2],
            "spam_probability": round(float(r[3]), 4),
            "inference_ms": round(float(r[4]), 3),
            "model_used": r[5],
            "source": r[6],
            "created_at": r[7].isoformat() if r[7] else None,
        }
        for r in rows
    ]


def save_benchmark_results(run_id: str, results: List[Dict[str, Any]]) -> None:
    """Persist Phase 3 and LSTM model benchmarking results into PostgreSQL."""
    if not is_db_configured():
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            for item in results:
                cur.execute(
                    """
                    INSERT INTO model_benchmarks (
                        run_id,
                        model_name,
                        accuracy,
                        precision_score,
                        recall_score,
                        f1_score,
                        confusion_matrix,
                        is_selected
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s);
                    """,
                    (
                        run_id,
                        item["model"],
                        float(item["accuracy"]),
                        float(item["precision"]),
                        float(item["recall"]),
                        float(item["f1"]),
                        json.dumps(item["confusion_matrix"]),
                        bool(item.get("selected", False)),
                    ),
                )
