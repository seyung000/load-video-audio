from pathlib import Path

import numpy as np

from config.connections import connect_db
from ingest.common import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_FRAMES_PER_CHUNK,
    DEFAULT_VIDEO_PATH,
    VideoChunk,
    cosine_similarity,
    ensure_embedding_dimension,
    extract_video_chunks,
    load_clip_model,
)
from utils.vector import vector_literal


DEFAULT_SIMILARITY_THRESHOLD = 0.97


def upsert_video(video_path: Path, duration_sec: int, chunk_seconds: int) -> int:
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
        connection.commit()
    return int(video_id)


def fetch_last_chunk(video_id: int) -> dict | None:
    with connect_db() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, chunk_index, start_sec, end_sec, embedding
                FROM video_chunks
                WHERE video_id = %s
                ORDER BY chunk_index DESC
                LIMIT 1
                """,
                (video_id,),
            )
            row = cursor.fetchone()

    if not row:
        return None

    embedding_raw = row[4]
    if isinstance(embedding_raw, str):
        embedding = np.fromstring(embedding_raw.strip("[]"), sep=",", dtype=np.float32)
    else:
        embedding = np.asarray(embedding_raw, dtype=np.float32)

    return {
        "id": int(row[0]),
        "chunk_index": int(row[1]),
        "start_sec": int(row[2]),
        "end_sec": int(row[3]),
        "embedding": embedding,
    }


def extend_existing_chunk(chunk_id: int, new_end_sec: int) -> None:
    with connect_db() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE video_chunks
                SET end_sec = %s
                WHERE id = %s
                """,
                (new_end_sec, chunk_id),
            )
        connection.commit()


def insert_scene_chunk(video_id: int, chunk: VideoChunk, chunk_index: int) -> None:
    with connect_db() as connection:
        with connection.cursor() as cursor:
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
                    chunk_index,
                    chunk.start_sec,
                    chunk.end_sec,
                    chunk.summary_text,
                    vector_literal(chunk.embedding),
                ),
            )
            cursor.execute(
                """
                INSERT INTO video_events (video_id, event_type, start_sec, end_sec, description)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    video_id,
                    "scene_change",
                    chunk.start_sec,
                    chunk.end_sec,
                    "New scene stored by fast ingest after CLIP similarity check.",
                ),
            )
        connection.commit()


def ingest_with_similarity_merge(video_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    device, processor, model = load_clip_model()

    # 1. 10초 청크 생성
    # 2. 각 청크에서 3프레임 샘플링
    # 3. 샘플링된 프레임으로 CLIP 임베딩 생성
    duration_sec, chunks = extract_video_chunks(
        video_path=video_path,
        chunk_seconds=DEFAULT_CHUNK_SECONDS,
        frames_per_chunk=DEFAULT_FRAMES_PER_CHUNK,
        processor=processor,
        model=model,
        device=device,
    )
    ensure_embedding_dimension(chunks, expected_dimension=model.config.projection_dim)

    video_id = upsert_video(video_path, duration_sec, DEFAULT_CHUNK_SECONDS)
    last_chunk = fetch_last_chunk(video_id)
    next_chunk_index = 0 if last_chunk is None else last_chunk["chunk_index"] + 1

    stored_count = 0
    merged_count = 0

    for chunk in chunks:
        if last_chunk is not None:
            # 4. 직전 저장 청크와 현재 청크의 유사도 비교
            similarity = cosine_similarity(last_chunk["embedding"], chunk.embedding)
            if similarity >= DEFAULT_SIMILARITY_THRESHOLD:
                # 5. 중복이면 새 row를 만들지 않고 기존 청크의 종료 시간만 갱신
                extend_existing_chunk(last_chunk["id"], chunk.end_sec)
                last_chunk["end_sec"] = chunk.end_sec
                merged_count += 1
                print(
                    f"merged {chunk.start_sec}-{chunk.end_sec}s into chunk_id={last_chunk['id']} "
                    f"(similarity={similarity:.4f})"
                )
                continue

        # 6. 새 장면이면 청크를 저장하고 scene_change 이벤트 생성
        insert_scene_chunk(video_id, chunk, next_chunk_index)
        print(
            f"stored chunk_index={next_chunk_index} time={chunk.start_sec}-{chunk.end_sec}s as new scene"
        )
        last_chunk = fetch_last_chunk(video_id)
        next_chunk_index += 1
        stored_count += 1

    print(
        f"fast_ingest finished video={video_path} duration={duration_sec}s "
        f"stored={stored_count} merged={merged_count} threshold={DEFAULT_SIMILARITY_THRESHOLD}"
    )


def main() -> None:
    ingest_with_similarity_merge(DEFAULT_VIDEO_PATH)


if __name__ == "__main__":
    main()
