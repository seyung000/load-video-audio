from config.connections import connect_db


DEFAULT_VIDEO_ID = 1
SHORT_TERM_SECONDS = 60
LONG_TERM_SECONDS = 120


def load_video_chunks(video_id: int) -> list[dict]:
    with connect_db() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_index, start_sec, end_sec, summary_text, embedding
                FROM video_chunks
                WHERE video_id = %s
                ORDER BY chunk_index
                """,
                (video_id,),
            )
            rows = cursor.fetchall()

    return [
        {
            "chunk_index": row[0],
            "start_sec": row[1],
            "end_sec": row[2],
            "summary_text": row[3],
            "embedding": row[4],
        }
        for row in rows
    ]


def load_video_events(video_id: int) -> list[dict]:
    with connect_db() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT start_sec, end_sec, event_type, description, metadata
                FROM video_events
                WHERE video_id = %s
                ORDER BY start_sec
                """,
                (video_id,),
            )
            rows = cursor.fetchall()

    return [
        {
            "start_sec": row[0],
            "end_sec": row[1],
            "event_type": row[2],
            "description": row[3],
            "metadata": row[4],
        }
        for row in rows
    ]


# 청크가 윈도우 경계에 걸치면 양쪽 윈도우 모두에 포함
def overlaps(start_sec: int, end_sec: int, window_start: int, window_end: int) -> bool:
    return start_sec < window_end and end_sec > window_start

# 윈도우 생성
def build_windows(last_end_sec: int, window_seconds: int) -> list[dict]:
    windows: list[dict] = []
    window_start = 0

    while window_start < last_end_sec:
        windows.append(
            {
                "start_sec": window_start,
                "end_sec": window_start + window_seconds,
                "chunks": [],
                "events": [],
            }
        )
        window_start += window_seconds

    return windows


def group_by_short_term(
    chunks: list[dict],
    events: list[dict],
    short_term_seconds: int = SHORT_TERM_SECONDS,
) -> list[dict]:
    if not chunks and not events:
        return []

    last_chunk_end = max((chunk["end_sec"] for chunk in chunks), default=0)
    last_event_end = max((event["end_sec"] for event in events), default=0)
    windows = build_windows(max(last_chunk_end, last_event_end), short_term_seconds)

    for window in windows:
        for chunk in chunks:
            if overlaps(chunk["start_sec"], chunk["end_sec"], window["start_sec"], window["end_sec"]):
                window["chunks"].append(chunk)

        for event in events:
            if overlaps(event["start_sec"], event["end_sec"], window["start_sec"], window["end_sec"]):
                window["events"].append(event)

    return [window for window in windows if window["chunks"] or window["events"]]


def group_by_long_term(
    short_terms: list[dict],
    long_term_seconds: int = LONG_TERM_SECONDS,
) -> list[dict]:
    if not short_terms:
        return []

    last_end_sec = max(short_term["end_sec"] for short_term in short_terms)
    windows = build_windows(last_end_sec, long_term_seconds)

    long_terms: list[dict] = []
    for window in windows:
        matched_short_terms = [
            short_term
            for short_term in short_terms
            if overlaps(short_term["start_sec"], short_term["end_sec"], window["start_sec"], window["end_sec"])
        ]
        if matched_short_terms:
            long_terms.append(
                {
                    "start_sec": window["start_sec"],
                    "end_sec": window["end_sec"],
                    "short_terms": matched_short_terms,
                }
            )

    return long_terms


def make_short_summary_text(short_term: dict) -> str:
    chunk_texts = []
    seen_chunk_texts: set[str] = set()
    for chunk in short_term["chunks"]:
        text = chunk["summary_text"]
        if text not in seen_chunk_texts:
            seen_chunk_texts.add(text)
            chunk_texts.append(text)

    event_texts = []
    for event in short_term["events"]:
        description = event["description"] or event["event_type"]
        event_texts.append(f"[{event['event_type']}] {description}")

    parts = []
    if chunk_texts:
        parts.append("Scenes: " + " ".join(chunk_texts))
    if event_texts:
        parts.append("Events: " + " ".join(event_texts))

    return " ".join(parts) if parts else "No significant activity."


def make_long_summary_text(long_term: dict) -> str:
    short_texts = [short_term["summary_text"] for short_term in long_term["short_terms"] if short_term["summary_text"]]
    if not short_texts:
        return "No significant long-term activity."
    return " ".join(short_texts)


def generate_short_term_summaries(short_terms: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    for short_term in short_terms:
        summaries.append(
            {
                "start_sec": short_term["start_sec"],
                "end_sec": short_term["end_sec"],
                "summary_text": make_short_summary_text(short_term),
                "metadata": {
                    "chunk_count": len(short_term["chunks"]),
                    "event_count": len(short_term["events"]),
                },
            }
        )
    return summaries


def generate_long_term_summaries(short_summaries: list[dict]) -> list[dict]:
    long_terms = group_by_long_term(short_summaries, LONG_TERM_SECONDS)
    summaries: list[dict] = []

    for long_term in long_terms:
        summary_text = make_long_summary_text(long_term)
        summaries.append(
            {
                "period_label": "test_window",
                "start_sec": long_term["start_sec"],
                "end_sec": long_term["end_sec"],
                "summary_text": summary_text,
                "metadata": {
                    "short_term_count": len(long_term["short_terms"]),
                },
            }
        )

    return summaries


def main() -> None:
    video_id = DEFAULT_VIDEO_ID
    chunks = load_video_chunks(video_id)
    events = load_video_events(video_id)
    short_terms = group_by_short_term(chunks, events)
    short_summaries = generate_short_term_summaries(short_terms)
    long_summaries = generate_long_term_summaries(short_summaries)

    print("Short-term summaries:")
    for summary in short_summaries:
        print(summary)

    print("\nLong-term summaries:")
    for summary in long_summaries:
        print(summary)


if __name__ == "__main__":
    main()
