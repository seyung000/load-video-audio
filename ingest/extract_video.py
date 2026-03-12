import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection

from config.connections import connect_db
from utils.device import get_device
from utils.vector import normalize, vector_literal


# 사용할 CLIP 모델 이름 (비전 전용 — 이미지 임베딩 생성)
DEFAULT_MODEL_NAME = "openai/clip-vit-large-patch14-336"
# 처리할 비디오 파일 경로
DEFAULT_VIDEO_PATH = Path("data/sample_video.mp4")
# 비디오를 나눌 청크 단위 (초)
DEFAULT_CHUNK_SECONDS = 10
# 청크당 샘플링할 프레임 수
DEFAULT_FRAMES_PER_CHUNK = 3
# True로 설정하면 DB 저장을 건너뜀 (임베딩 결과만 확인할 때 사용)
DEFAULT_SKIP_DB = False


@dataclass
class VideoChunk:
    """비디오 한 청크의 메타데이터와 임베딩을 담는 자료구조"""
    chunk_index: int        # 청크 순번
    start_sec: int          # 청크 시작 시각 (초)
    end_sec: int            # 청크 종료 시각 (초)
    frame_count: int        # 실제로 추출된 프레임 수
    embedding: np.ndarray   # CLIP으로 생성된 벡터 (768차원)

    @property
    def summary_text(self) -> str:
        # DB에 저장할 설명 텍스트 — 현재는 형식적인 문자열
        # 추후 멀티모달 캡셔닝 모델로 실제 장면 설명으로 대체 예정
        return (
            f"Visual-only chunk embedding generated from {self.frame_count} sampled frames "
            f"between {self.start_sec}s and {self.end_sec}s."
        )



def build_frame_timestamps(start_sec: int, end_sec: int, frames_per_chunk: int) -> list[float]:
    """
    청크 구간 내에서 균등하게 프레임을 샘플링할 타임스탬프 목록을 생성.
    예) 0~10초 구간에서 3프레임 → [2.5, 5.0, 7.5]초
    """
    if frames_per_chunk <= 0:
        raise ValueError("frames_per_chunk must be greater than 0.")
    span = max(end_sec - start_sec, 1)
    return [
        start_sec + (span * (index + 1) / (frames_per_chunk + 1))
        for index in range(frames_per_chunk)
    ]


def read_frames(capture: cv2.VideoCapture, timestamps: Iterable[float], fps: float) -> list[np.ndarray]:
    """
    지정된 타임스탬프에 해당하는 프레임을 OpenCV로 추출.
    BGR → RGB 변환 후 반환 (CLIP 모델 입력 포맷에 맞춤)
    """
    frames: list[np.ndarray] = []
    for timestamp in timestamps:
        # 타임스탬프(초) → 프레임 인덱스 변환
        frame_index = max(int(timestamp * fps), 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        # OpenCV 기본 포맷(BGR)을 CLIP 입력 포맷(RGB)으로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
    return frames


def embed_frames(
    frames: list[np.ndarray],
    processor: AutoProcessor,
    model: CLIPVisionModelWithProjection,
    device: torch.device,
) -> np.ndarray:
    """
    프레임 목록을 CLIP 모델에 통과시켜 청크 단위 임베딩을 생성.
    여러 프레임의 임베딩을 평균 → L2 정규화하여 청크를 대표하는 단일 벡터로 압축.
    """
    # 프레임 전처리 (리사이즈, 정규화 등)
    inputs = processor(images=frames, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        # 각 프레임의 임베딩 벡터 추출 (shape: [프레임 수, 768])
        image_embeds = outputs.image_embeds.detach().cpu().numpy()

    # 프레임 임베딩을 평균 내어 청크를 대표하는 벡터 하나로 합침 → L2 정규화
    pooled = normalize(image_embeds.mean(axis=0))
    return pooled.astype(np.float32)


def extract_video_chunks(
    video_path: Path,
    chunk_seconds: int,
    frames_per_chunk: int,
    processor: AutoProcessor,
    model: CLIPVisionModelWithProjection,
    device: torch.device,
) -> tuple[int, list[VideoChunk]]:
    """
    비디오 전체를 chunk_seconds 단위로 나누고,
    각 청크에서 프레임을 샘플링해 CLIP 임베딩을 생성하여 VideoChunk 리스트로 반환.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        capture.release()
        raise RuntimeError("Failed to read FPS or frame count from the video.")

    # 전체 재생 시간 (초) 계산
    duration_sec = math.ceil(total_frames / fps)
    chunks: list[VideoChunk] = []

    try:
        chunk_count = math.ceil(duration_sec / chunk_seconds)
        for chunk_index in range(chunk_count):
            start_sec = chunk_index * chunk_seconds
            end_sec = min((chunk_index + 1) * chunk_seconds, duration_sec)

            # 1. 청크 내 샘플링 타임스탬프 계산
            timestamps = build_frame_timestamps(start_sec, end_sec, frames_per_chunk)
            # 2. 해당 타임스탬프의 프레임 추출
            frames = read_frames(capture, timestamps, fps)
            if not frames:
                continue
            # 3. 프레임들을 CLIP으로 임베딩 → 청크 대표 벡터 생성
            embedding = embed_frames(frames, processor, model, device)
            chunks.append(
                VideoChunk(
                    chunk_index=chunk_index,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    frame_count=len(frames),
                    embedding=embedding,
                )
            )
    finally:
        # 예외 발생 여부와 무관하게 VideoCapture 리소스 해제
        capture.release()

    return duration_sec, chunks


def ensure_embedding_dimension(chunks: list[VideoChunk], expected_dimension: int = 768) -> None:
    """
    모든 청크의 임베딩 차원이 DB 스키마와 일치하는지 검증.
    모델을 교체할 경우 infra/init.sql의 벡터 차원도 함께 수정해야 함.
    """
    for chunk in chunks:
        actual_dimension = int(chunk.embedding.shape[0])
        if actual_dimension != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dimension}, got {actual_dimension}. "
                "Update infra/init.sql if you switch models."
            )


def upsert_video_and_chunks(video_path: Path, duration_sec: int, chunk_seconds: int, chunks: list[VideoChunk]) -> None:
    """
    비디오 메타데이터와 청크 임베딩을 PostgreSQL에 저장 (upsert).
    같은 file_path로 재실행하면 덮어씀 (ON CONFLICT DO UPDATE).
    """
    with connect_db() as connection:
        with connection.cursor() as cursor:
            # 1. videos 테이블에 비디오 메타데이터 upsert → 생성된 video_id 획득
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

            # 2. video_chunks 테이블에 각 청크 임베딩 upsert
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
                        vector_literal(chunk.embedding),  # pgvector 문자열 포맷으로 변환
                    ),
                )
        connection.commit()


def main() -> None:
    # 비디오 파일 존재 여부 확인
    video_path = DEFAULT_VIDEO_PATH
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 디바이스 및 CLIP 모델 로드
    device = get_device()
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_NAME)
    model = CLIPVisionModelWithProjection.from_pretrained(DEFAULT_MODEL_NAME).to(device)
    model.eval()

    # 비디오 → 청크 임베딩 생성
    duration_sec, chunks = extract_video_chunks(
        video_path=video_path,
        chunk_seconds=DEFAULT_CHUNK_SECONDS,
        frames_per_chunk=DEFAULT_FRAMES_PER_CHUNK,
        processor=processor,
        model=model,
        device=device,
    )

    # 모델의 실제 projection 차원과 임베딩 차원이 일치하는지 검증
    ensure_embedding_dimension(chunks, expected_dimension=model.config.projection_dim)

    # 처리 결과 출력
    print(f"video={video_path} duration={duration_sec}s chunks={len(chunks)} device={device}")
    for chunk in chunks:
        print(
            f"chunk={chunk.chunk_index} time={chunk.start_sec}-{chunk.end_sec}s "
            f"frames={chunk.frame_count} dim={chunk.embedding.shape[0]}"
        )

    # DEFAULT_SKIP_DB=True이면 DB 저장 없이 종료 (임베딩 확인용)
    if DEFAULT_SKIP_DB:
        return

    # PostgreSQL에 임베딩 저장
    upsert_video_and_chunks(video_path, duration_sec, DEFAULT_CHUNK_SECONDS, chunks)
    print("Stored chunk embeddings in PostgreSQL.")


if __name__ == "__main__":
    main()
