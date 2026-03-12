CREATE EXTENSION IF NOT EXISTS vector;

-- videos table
CREATE TABLE IF NOT EXISTS videos (
    id BIGSERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    duration_sec INTEGER NOT NULL CHECK (duration_sec >= 0),
    chunk_seconds INTEGER NOT NULL DEFAULT 10 CHECK (chunk_seconds > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- video_chunks table
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

-- video events table
CREATE TABLE IF NOT EXISTS video_events (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    start_sec INTEGER NOT NULL CHECK (start_sec >= 0),
    end_sec INTEGER NOT NULL CHECK (end_sec >= start_sec),
    description TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- short term video summaries
CREATE TABLE IF NOT EXISTS video_short_summaries (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start_sec INTEGER NOT NULL CHECK (start_sec >= 0),
    end_sec INTEGER NOT NULL CHECK (end_sec > start_sec),
    summary_text TEXT NOT NULL,
    embedding VECTOR(768),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (video_id, start_sec, end_sec)
);

-- long term video summaries
CREATE TABLE IF NOT EXISTS video_long_summaries (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    period_label TEXT NOT NULL,
    start_sec INTEGER NOT NULL CHECK (start_sec >= 0),
    end_sec INTEGER NOT NULL CHECK (end_sec > start_sec),
    summary_text TEXT NOT NULL,
    embedding VECTOR(768),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (video_id, period_label, start_sec, end_sec)
);


CREATE INDEX IF NOT EXISTS idx_videos_created_at
    ON videos (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_video_chunks_video_id_chunk_index
    ON video_chunks (video_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_video_chunks_video_id_start_sec
    ON video_chunks (video_id, start_sec);

CREATE INDEX IF NOT EXISTS idx_video_events_video_id_start_sec
    ON video_events (video_id, start_sec);

CREATE INDEX IF NOT EXISTS idx_video_events_event_type
    ON video_events (event_type);

CREATE INDEX IF NOT EXISTS idx_video_short_summaries_video_id_start_sec
    ON video_short_summaries (video_id, start_sec);

CREATE INDEX IF NOT EXISTS idx_video_long_summaries_video_id_start_sec
    ON video_long_summaries (video_id, start_sec);

COMMENT ON TABLE videos IS 'Original uploaded video metadata.';
COMMENT ON TABLE video_chunks IS '10-second video chunks with visual summaries and embeddings.';
COMMENT ON TABLE video_events IS 'Detected events such as motion, object appearance, person entry, or speech activity.';
COMMENT ON TABLE video_short_summaries IS 'Short-term rolling summaries used for recent-context retrieval.';
COMMENT ON TABLE video_long_summaries IS 'Long-term summaries for larger time windows such as sessions, hours, or a full day.';
COMMENT ON COLUMN video_chunks.summary_text IS 'Chunk-level visual summary or debug text for semantic search.';
COMMENT ON COLUMN video_chunks.embedding IS 'CLIP visual embedding. Update dimension if the embedding model changes.';
COMMENT ON COLUMN video_events.metadata IS 'Structured event metadata such as detected objects, confidence, or motion score.';
COMMENT ON COLUMN video_short_summaries.metadata IS 'Structured metadata for recent summaries.';
COMMENT ON COLUMN video_short_summaries.embedding IS 'Embedding for semantic retrieval over short-term summaries.';
COMMENT ON COLUMN video_long_summaries.period_label IS 'A label such as hourly, session, daily, or custom.';
COMMENT ON COLUMN video_long_summaries.metadata IS 'Structured metadata for long-term summaries.';
COMMENT ON COLUMN video_long_summaries.embedding IS 'Embedding for semantic retrieval over long-term summaries.';
