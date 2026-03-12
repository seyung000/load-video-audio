import numpy as np
import torch
from transformers import AutoProcessor, CLIPTextModelWithProjection

from config.connections import connect_db
from config.llm_client import get_model
from utils.device import get_device
from utils.vector import normalize, vector_literal


DEFAULT_MODEL_NAME = "openai/clip-vit-large-patch14-336"
DEFAULT_TOP_K = 5
SHORT_TERM_DISTANCE_THRESHOLD = 0.30
LONG_TERM_DISTANCE_THRESHOLD = 0.35


def embed_text(text: str, processor: AutoProcessor, model: CLIPTextModelWithProjection, device: torch.device) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        embed = model(**inputs).text_embeds.detach().cpu().numpy()[0]
    return normalize(embed).astype(np.float32)


def fetch_similar_rows(cur, table_name: str, alias: str, query_vector: np.ndarray, top_k: int) -> list[dict]:
    cur.execute(
        f"""
        SELECT v.file_name, {alias}.start_sec, {alias}.end_sec,
               {alias}.embedding <=> CAST(%s AS vector) AS distance
        FROM {table_name} {alias}
        JOIN videos v ON v.id = {alias}.video_id
        ORDER BY distance ASC
        LIMIT %s
        """,
        (vector_literal(query_vector), top_k),
    )
    return [
        {"file_name": row[0], "start_sec": row[1], "end_sec": row[2], "distance": float(row[3])}
        for row in cur.fetchall()
    ]

# 임계값 만족 여부 확인
def is_good_enough(results: list[dict], threshold: float) -> bool:
    return bool(results) and results[0]["distance"] <= threshold


def search_chunks(query_vector: np.ndarray, top_k: int) -> list[dict]:
    with connect_db() as conn:
        with conn.cursor() as cur:
            short_results = fetch_similar_rows(cur, "video_short_summaries", "vs", query_vector, top_k)
            if is_good_enough(short_results, SHORT_TERM_DISTANCE_THRESHOLD):
                return short_results

            long_results = fetch_similar_rows(cur, "video_long_summaries", "vl", query_vector, top_k)
            if is_good_enough(long_results, LONG_TERM_DISTANCE_THRESHOLD):
                return long_results

            return fetch_similar_rows(cur, "video_chunks", "vc", query_vector, top_k)


def answer_with_context(user_query: str, chunks: list[dict]) -> str:
    context = "\n".join(
        f"- {chunk['file_name']} {chunk['start_sec']}s~{chunk['end_sec']}s (distance: {chunk['distance']:.4f})"
        for chunk in chunks
    )
    client, model = get_model()
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    f"The list of scenes visually similar to your question is:\n\n{context}\n\n"
                    f"Question: {user_query}\n\nPlease answer based on the scene information above."
                ),
            }
        ],
    )
    return response.choices[0].message.content


def main() -> None:
    device = get_device()
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_NAME)
    model = CLIPTextModelWithProjection.from_pretrained(DEFAULT_MODEL_NAME).to(device)
    model.eval()

    print("Enter the question: (Q to quit)")
    while True:
        user_query = input("\n질문: ").strip()
        if user_query.lower() == "q":
            break
        if not user_query:
            continue

        query_vector = embed_text(user_query, processor, model, device)
        chunks = search_chunks(query_vector, top_k=DEFAULT_TOP_K)
        if not chunks:
            print("The search results are not found.")
            continue

        answer = answer_with_context(user_query, chunks)
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
