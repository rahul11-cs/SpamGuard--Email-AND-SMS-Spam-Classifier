-- Migration: 001_add_lstm_embedding_hnsw_index
-- Purpose: Create an HNSW vector index on classification_logs.lstm_embedding
--          to optimize pgvector cosine similarity (<=>) searches.
-- Target PostgreSQL + pgvector >= 0.5.0

-- Ensure pgvector extension is enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Ensure target column exists on pre-LSTM deployments
ALTER TABLE classification_logs
    ADD COLUMN IF NOT EXISTS lstm_embedding VECTOR(128);

-- Create HNSW index using cosine distance operator class (matches `<=>` queries).
-- Tuning parameters:
--   m = 16               : max bi-directional links per node (balanced recall/memory)
--   ef_construction = 64 : dynamic candidate list size during graph construction
-- Note: For zero-downtime indexing on large live production tables, run outside a
-- transaction block with CONCURRENTLY:
--   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_classification_logs_lstm_embedding_hnsw
--       ON classification_logs
--       USING hnsw (lstm_embedding vector_cosine_ops)
--       WITH (m = 16, ef_construction = 64)
--       WHERE lstm_embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_classification_logs_lstm_embedding_hnsw
    ON classification_logs
    USING hnsw (lstm_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE lstm_embedding IS NOT NULL;

-- Refresh planner statistics so PostgreSQL selects the HNSW index for ORDER BY <=> LIMIT queries
ANALYZE classification_logs;
