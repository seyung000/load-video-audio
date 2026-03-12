import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection

from utils.device import get_device
from utils.vector import normalize


# 기본 상수
DEFAULT_MODEL_NAME = "openai/clip-vit-large-patch14-336"
DEFAULT_VIDEO_PATH = Path("data/sample_video.mp4")
DEFAULT_CHUNK_SECONDS = 10       # 청크 단위 길이 (초)
DEFAULT_FRAMES_PER_CHUNK = 3     # 청크당 샘플링할 프레임 수
DEFAULT_SKIP_DB = False          # True면 DB 저장 건너뜀


@dataclass
class VideoChunk:
    chunk_index: int        # 청크 순서 번호
    start_sec: int          # 청크 시작 시간 (초)
    end_sec: int            # 청크 종료 시간 (초)
    frame_count: int        # 실제 읽은 프레임 수
    embedding: np.ndarray   # CLIP 임베딩 벡터

    @property
    def summary_text(self) -> str:
        return (
            f"Visual-only chunk embedding generated from {self.frame_count} sampled frames "
            f"between {self.start_sec}s and {self.end_sec}s."
        )


def build_frame_timestamps(start_sec: int, end_sec: int, frames_per_chunk: int) -> list[float]:
    """청크 구간 내에서 균등 간격으로 프레임 타임스탬프를 생성한다."""
    if frames_per_chunk <= 0:
        raise ValueError("frames_per_chunk must be greater than 0.")
    span = max(end_sec - start_sec, 1)
    # 구간을 (frames_per_chunk + 1) 등분하여 양 끝단을 제외한 내부 지점만 추출
    return [
        start_sec + (span * (index + 1) / (frames_per_chunk + 1))
        for index in range(frames_per_chunk)
    ]


def read_frames(capture: cv2.VideoCapture, timestamps: Iterable[float], fps: float) -> list[np.ndarray]:
    """타임스탬프 목록에 해당하는 프레임을 읽어 RGB 배열로 반환한다."""
    frames: list[np.ndarray] = []
    for timestamp in timestamps:
        frame_index = max(int(timestamp * fps), 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
    return frames


def embed_frames(
    frames: list[np.ndarray],
    processor: AutoProcessor,
    model: CLIPVisionModelWithProjection,
    device: torch.device,
) -> np.ndarray:
    """프레임 목록을 CLIP 모델에 통과시켜 정규화된 단일 임베딩 벡터를 반환한다.

    여러 프레임의 임베딩을 평균낸 뒤 L2 정규화하여 코사인 유사도 비교에 바로 사용할 수 있게 한다.
    """
    inputs = processor(images=frames, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds.detach().cpu().numpy()

    # 프레임별 임베딩 평균 → L2 정규화
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
    """비디오 파일을 chunk_seconds 단위로 분할하고 각 청크의 CLIP 임베딩을 생성한다."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        capture.release()
        raise RuntimeError("Failed to read FPS or frame count from the video.")

    duration_sec = math.ceil(total_frames / fps)
    chunks: list[VideoChunk] = []

    try:
        chunk_count = math.ceil(duration_sec / chunk_seconds)
        for chunk_index in range(chunk_count):
            start_sec = chunk_index * chunk_seconds
            end_sec = min((chunk_index + 1) * chunk_seconds, duration_sec)
            timestamps = build_frame_timestamps(start_sec, end_sec, frames_per_chunk)
            frames = read_frames(capture, timestamps, fps)
            if not frames:
                continue
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
        capture.release()

    return duration_sec, chunks


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """두 임베딩 벡터의 코사인 유사도를 반환한다 (범위: -1 ~ 1)."""
    left_norm = normalize(left)
    right_norm = normalize(right)
    return float(np.dot(left_norm, right_norm))


def ensure_embedding_dimension(chunks: list[VideoChunk], expected_dimension: int = 768) -> None:
    """모든 청크의 임베딩 차원이 DB 스키마와 일치하는지 검증한다.

    모델을 교체할 경우 infra/init.sql의 벡터 차원도 함께 수정해야 한다.
    """
    for chunk in chunks:
        actual_dimension = int(chunk.embedding.shape[0])
        if actual_dimension != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dimension}, got {actual_dimension}. "
                "Update infra/init.sql if you switch models."
            )


def load_clip_model() -> tuple[torch.device, AutoProcessor, CLIPVisionModelWithProjection]:
    """CLIP 모델과 프로세서를 로드하고 추론 모드로 설정한다."""
    device = get_device()
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_NAME)
    model = CLIPVisionModelWithProjection.from_pretrained(DEFAULT_MODEL_NAME).to(device)
    model.eval()
    return device, processor, model
