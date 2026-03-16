from config.connections import connect_db
from ingest.common import DEFAULT_MODEL_NAME
from memory.make_summary import (
    DEFAULT_VIDEO_ID,
    generate_long_term_summaries,
    generate_short_term_summaries,
    group_by_short_term,
    load_video_chunks,
    load_video_events,
)
from utils.device import get_device
from utils.vector import normalize, vector_literal

import numpy as np
from psycopg.types.json import Jsonb
import torch
from transformers import AutoProcessor, CLIPTextModelWithProjection


def load_text_embedding_model() -> tuple[torch.device, AutoProcessor, CLIPTextModelWithProjection]:
    device = get_device()
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_NAME)
    model = CLIPTextModelWithProjection.from_pretrained(DEFAULT_MODEL_NAME).to(device)
    model.eval()
    return device, processor, model


def embed_text(
    text: str,
    processor: AutoProcessor,
    model: CLIPTextModelWithProjection,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(device)
    with torch.inference_mode():
        embedding = model(**inputs).text_embeds.detach().cpu().numpy()[0]
    return normalize(embedding).astype(np.float32)


def upsert_short_term_memories(
    video_id: int,
    summaries: list[dict],
    processor: AutoProcessor,
    model: CLIPTextModelWithProjection,
    device: torch.device,
) -> None:
    with connect_db() as connection:
        with connection.cursor() as cursor:
            for summary in summaries:
                embedding = embed_text(summary["summary_text"], processor, model, device)
                cursor.execute(
                    """
                    INSERT INTO video_short_summaries (
                        video_id,
                        start_sec,
                        end_sec,
                        summary_text,
                        embedding,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, CAST(%s AS vector), %s)
                    ON CONFLICT (video_id, start_sec, end_sec)
                    DO UPDATE SET
                        summary_text = EXCLUDED.summary_text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        video_id,
                        summary["start_sec"],
                        summary["end_sec"],
                        summary["summary_text"],
                        vector_literal(embedding),
                        Jsonb(summary["metadata"]),
                    ),
                )
        connection.commit()


def upsert_long_term_memories(
    video_id: int,
    summaries: list[dict],
    processor: AutoProcessor,
    model: CLIPTextModelWithProjection,
    device: torch.device,
) -> None:
    with connect_db() as connection:
        with connection.cursor() as cursor:
            for summary in summaries:
                embedding = embed_text(summary["summary_text"], processor, model, device)
                cursor.execute(
                    """
                    INSERT INTO video_long_summaries (
                        video_id,
                        period_label,
                        start_sec,
                        end_sec,
                        summary_text,
                        embedding,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, CAST(%s AS vector), %s)
                    ON CONFLICT (video_id, period_label, start_sec, end_sec)
                    DO UPDATE SET
                        summary_text = EXCLUDED.summary_text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        video_id,
                        summary["period_label"],
                        summary["start_sec"],
                        summary["end_sec"],
                        summary["summary_text"],
                        vector_literal(embedding),
                        Jsonb(summary["metadata"]),
                    ),
                )
        connection.commit()


def build_memories(video_id: int) -> tuple[list[dict], list[dict]]:
    chunks = load_video_chunks(video_id)
    events = load_video_events(video_id)
    short_terms = group_by_short_term(chunks, events)
    short_summaries = generate_short_term_summaries(short_terms)
    long_summaries = generate_long_term_summaries(short_summaries)

    device, processor, model = load_text_embedding_model()
    upsert_short_term_memories(video_id, short_summaries, processor, model, device)
    upsert_long_term_memories(video_id, long_summaries, processor, model, device)

    return short_summaries, long_summaries


def main() -> None:
    short_summaries, long_summaries = build_memories(DEFAULT_VIDEO_ID)
    print(f"short-term summaries stored: {len(short_summaries)}")
    print(f"long-term summaries stored: {len(long_summaries)}")


if __name__ == "__main__":
    main()
