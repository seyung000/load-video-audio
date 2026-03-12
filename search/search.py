
import numpy as np
import torch
from transformers import AutoProcessor, CLIPTextModelWithProjection

from config.connections import connect_db
from config.llm_client import get_model
from utils.device import get_device
from utils.vector import normalize, vector_literal


# ingest/extract_video.py와 동일한 모델 (텍스트-이미지가 같은 벡터 공간에 매핑되어야 검색 가능)
DEFAULT_MODEL_NAME = "openai/clip-vit-large-patch14-336"
DEFAULT_TOP_K = 5


# def make_search_query(user_query: str) -> str:
#     """
#     사용자 질문을 CLIP 검색에 적합한 영어 장면 묘사로 변환.
#     CLIP은 영어 학습 데이터 기반이므로 영어 검색어가 유사도 정확도가 높음.
#     """
#     client, model = get_model()
#     response = client.chat.completions.create(
#         model=model,
#         max_tokens=64,
#         messages=[
#             {
#                 "role": "user",
#                 "content": (
#                     f"다음 질문을 CLIP 이미지 검색에 적합한 짧은 영어 장면 묘사로 변환해줘. "
#                     f"묘사 문장만 출력해.\n\n질문: {user_query}"
#                 ),
#             }
#         ],
#     )
#     return response.choices[0].message.content.strip()


def embed_text(text: str, processor: AutoProcessor, model: CLIPTextModelWithProjection, device: torch.device) -> np.ndarray:
    """텍스트 → CLIP 임베딩 → L2 정규화 (적재 시 이미지 임베딩과 동일한 공간)"""
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        embed = model(**inputs).text_embeds.detach().cpu().numpy()[0]
    return normalize(embed).astype(np.float32)


def search_chunks(query_vector: np.ndarray, top_k: int) -> list[dict]:
    """pgvector <=> (코사인 거리)로 유사 장면 청크 검색. 값이 작을수록 유사."""
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT v.file_name, vc.start_sec, vc.end_sec,
                       vc.embedding <=> CAST(%s AS vector) AS distance
                FROM video_chunks vc
                JOIN videos v ON v.id = vc.video_id
                ORDER BY distance ASC
                LIMIT %s
                """,
                (vector_literal(query_vector), top_k),
            )
            return [
                {"file_name": r[0], "start_sec": r[1], "end_sec": r[2], "distance": float(r[3])}
                for r in cur.fetchall()
            ]


def answer_with_context(user_query: str, chunks: list[dict]) -> str:
    """검색된 청크(시간대 정보)를 컨텍스트로 LLM에 전달해 최종 답변 생성."""
    context = "\n".join(
        f"- {c['file_name']} {c['start_sec']}s~{c['end_sec']}s (거리: {c['distance']:.4f})"
        for c in chunks
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

        # 1. 사용자 질문 → CLIP 검색용 영어 장면 묘사로 변환
        # search_query = make_search_query(user_query)
        # print(f"검색어: {search_query}")

        # 2. 검색어 임베딩 → pgvector 유사 청크 검색
        # query_vector = embed_text(search_query, processor, model, device)
        query_vector = embed_text(user_query, processor, model, device)
        chunks = search_chunks(query_vector, top_k=DEFAULT_TOP_K)
        if not chunks:
            print("The search results are not found.")
            continue

        # 3. 검색 결과를 컨텍스트로 LLM 최종 답변 생성
        answer = answer_with_context(user_query, chunks)
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
