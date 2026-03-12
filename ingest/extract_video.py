# Extract video chunks, generate embeddings, and store in PostgreSQL
# CLIP 모델 사용 테스트
from pathlib import Path

from config.connections import connect_db
from ingest.common import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_FRAMES_PER_CHUNK,
    DEFAULT_SKIP_DB,
    DEFAULT_VIDEO_PATH,
    VideoChunk,
    ensure_embedding_dimension,
    extract_video_chunks,
    load_clip_model,
)
from utils.vector import vector_literal


def upsert_video_and_chunks(video_path: Path, duration_sec: int, chunk_seconds: int, chunks: list[VideoChunk]) -> None:
    with connect_db() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO videos (file_name, file_path, duration_sec, chunk_seconds)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (file_path)
                DO UPDATE SET
                    file_name = EXCLUDED.file_name,
                    duration_sec = EXCLUDED.duration_sec,
                    chunk_seconds = EXCLUDED.chunk_seconds
                RETURNING id
                """,
                (video_path.name, str(video_path.resolve()), duration_sec, chunk_seconds),
            )
            video_id = cursor.fetchone()[0]

            for chunk in chunks:
                cursor.execute(
                    """
                    INSERT INTO video_chunks (
                        video_id,
                        chunk_index,
                        start_sec,
                        end_sec,
                        summary_text,
                        embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, CAST(%s AS vector))
                    ON CONFLICT (video_id, chunk_index)
                    DO UPDATE SET
                        start_sec = EXCLUDED.start_sec,
                        end_sec = EXCLUDED.end_sec,
                        summary_text = EXCLUDED.summary_text,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        video_id,
                        chunk.chunk_index,
                        chunk.start_sec,
                        chunk.end_sec,
                        chunk.summary_text,
                        vector_literal(chunk.embedding),
                    ),
                )
        connection.commit()


def main() -> None:
    video_path = DEFAULT_VIDEO_PATH
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    device, processor, model = load_clip_model()

    duration_sec, chunks = extract_video_chunks(
        video_path=video_path,
        chunk_seconds=DEFAULT_CHUNK_SECONDS,
        frames_per_chunk=DEFAULT_FRAMES_PER_CHUNK,
        processor=processor,
        model=model,
        device=device,
    )
    ensure_embedding_dimension(chunks, expected_dimension=model.config.projection_dim)

    print(f"video={video_path} duration={duration_sec}s chunks={len(chunks)} device={device}")
    for chunk in chunks:
        print(
            f"chunk={chunk.chunk_index} time={chunk.start_sec}-{chunk.end_sec}s "
            f"frames={chunk.frame_count} dim={chunk.embedding.shape[0]}"
        )

    if DEFAULT_SKIP_DB:
        return

    upsert_video_and_chunks(video_path, duration_sec, DEFAULT_CHUNK_SECONDS, chunks)
    print("Stored chunk embeddings in PostgreSQL.")


if __name__ == "__main__":
    main()
