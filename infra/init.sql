CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS videos (
    id BIGSERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    duration_sec INTEGER NOT NULL CHECK (duration_sec >= 0),
    chunk_seconds INTEGER NOT NULL DEFAULT 10 CHECK (chunk_seconds > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS video_chunks (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    start_sec INTEGER NOT NULL CHECK (start_sec >= 0),
    end_sec INTEGER NOT NULL CHECK (end_sec > start_sec),
    summary_text TEXT NOT NULL,
    embedding VECTOR(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (video_id, chunk_index),
    UNIQUE (video_id, start_sec, end_sec)
);

CREATE INDEX IF NOT EXISTS idx_videos_created_at
    ON videos (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_video_chunks_video_id_chunk_index
    ON video_chunks (video_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_video_chunks_video_id_start_sec
    ON video_chunks (video_id, start_sec);

COMMENT ON TABLE videos IS 'Original uploaded video metadata.';
COMMENT ON TABLE video_chunks IS '10-second video chunks with visual summaries and embeddings.';
COMMENT ON COLUMN video_chunks.summary_text IS 'Chunk-level visual summary or debug text for semantic search.';
COMMENT ON COLUMN video_chunks.embedding IS 'CLIP visual embedding. Update dimension if the embedding model changes.';
