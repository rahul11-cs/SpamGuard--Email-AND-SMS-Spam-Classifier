-- Rollback for 001_add_lstm_embedding_hnsw_index
-- Drops the HNSW vector index on classification_logs.lstm_embedding without removing column data.

DROP INDEX IF EXISTS idx_classification_logs_lstm_embedding_hnsw;
