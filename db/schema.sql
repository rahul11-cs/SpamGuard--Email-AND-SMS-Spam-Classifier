-- PostgreSQL + pgvector schema for Email Spam Classification Platform

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS classification_logs (
    id BIGSERIAL PRIMARY KEY,
    raw_text TEXT NOT NULL,
    transformed_text TEXT NOT NULL,
    prediction SMALLINT NOT NULL CHECK (prediction IN (0, 1)),
    prediction_label VARCHAR(16) NOT NULL CHECK (prediction_label IN ('Not Spam', 'Spam')),
    spam_probability DOUBLE PRECISION NOT NULL,
    inference_ms DOUBLE PRECISION NOT NULL,
    model_used VARCHAR(32) NOT NULL DEFAULT 'MultinomialNB',
    tfidf_vector VECTOR(3000),
    lstm_embedding VECTOR(128),
    source VARCHAR(32) NOT NULL DEFAULT 'streamlit',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Safe migration if table was created before LSTM support
ALTER TABLE classification_logs
    ALTER COLUMN tfidf_vector DROP NOT NULL;
ALTER TABLE classification_logs
    ADD COLUMN IF NOT EXISTS model_used VARCHAR(32) NOT NULL DEFAULT 'MultinomialNB';
ALTER TABLE classification_logs
    ADD COLUMN IF NOT EXISTS lstm_embedding VECTOR(128);

CREATE INDEX IF NOT EXISTS idx_classification_logs_created_at
    ON classification_logs (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_classification_logs_prediction
    ON classification_logs (prediction);

CREATE INDEX IF NOT EXISTS idx_classification_logs_lstm_embedding_hnsw
    ON classification_logs
    USING hnsw (lstm_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE lstm_embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS model_benchmarks (
    id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,
    model_name VARCHAR(64) NOT NULL,
    accuracy DOUBLE PRECISION NOT NULL,
    precision_score DOUBLE PRECISION NOT NULL,
    recall_score DOUBLE PRECISION NOT NULL,
    f1_score DOUBLE PRECISION NOT NULL,
    confusion_matrix JSONB NOT NULL,
    is_selected BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_benchmarks_run_id
    ON model_benchmarks (run_id, created_at DESC);
