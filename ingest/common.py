import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, CLIPModel

from utils.device import get_device
from utils.vector import normalize


DEFAULT_MODEL_NAME = "openai/clip-vit-large-patch14-336"
DEFAULT_VIDEO_PATH = Path("data/sample_video.mp4")
DEFAULT_CHUNK_SECONDS = 10
DEFAULT_FRAMES_PER_CHUNK = 3
DEFAULT_SKIP_DB = False
SCENE_LABELS = [
    "a person indoors",
    "people talking",
    "a person walking",
    "a person sitting at a desk",
    "a person using a laptop",
    "a person looking at objects",
    "an empty room",
    "an office scene",
    "a hallway or corridor",
    "a close-up of objects",
]


@dataclass
class VideoChunk:
    chunk_index: int
    start_sec: int
    end_sec: int
    frame_count: int
    summary_text: str
    embedding: np.ndarray


def build_frame_timestamps(start_sec: int, end_sec: int, frames_per_chunk: int) -> list[float]:
    if frames_per_chunk <= 0:
        raise ValueError("frames_per_chunk must be greater than 0.")
    span = max(end_sec - start_sec, 1)
    return [
        start_sec + (span * (index + 1) / (frames_per_chunk + 1))
        for index in range(frames_per_chunk)
    ]


def read_frames(capture: cv2.VideoCapture, timestamps: Iterable[float], fps: float) -> list[np.ndarray]:
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
    model: CLIPModel,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=frames, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        image_embeds = model.get_image_features(**inputs).detach().cpu().numpy()

    pooled = normalize(image_embeds.mean(axis=0))
    return pooled.astype(np.float32)


def summarize_frames(
    frames: list[np.ndarray],
    start_sec: int,
    end_sec: int,
    processor: AutoProcessor,
    model: CLIPModel,
    device: torch.device,
) -> str:
    mean_brightness = float(np.mean([frame.mean() for frame in frames]))

    if len(frames) >= 2:
        frame_diffs = [
            float(np.mean(np.abs(frames[index].astype(np.float32) - frames[index - 1].astype(np.float32))))
            for index in range(1, len(frames))
        ]
        motion_score = float(np.mean(frame_diffs))
    else:
        motion_score = 0.0

    text_inputs = processor(
        text=SCENE_LABELS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    image_inputs = processor(images=frames, return_tensors="pt")
    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}

    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)
        image_features = model.get_image_features(**image_inputs)

    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    image_features = torch.nn.functional.normalize(image_features, dim=-1)
    similarity = image_features @ text_features.T
    label_scores = similarity.mean(dim=0)
    best_label = SCENE_LABELS[int(torch.argmax(label_scores).item())]

    if motion_score < 8:
        motion_description = "with little visible movement"
    elif motion_score < 18:
        motion_description = "with some visible movement"
    else:
        motion_description = "with noticeable movement"

    if mean_brightness < 70:
        light_description = "dark"
    elif mean_brightness < 150:
        light_description = "moderately lit"
    else:
        light_description = "bright"

    return f"A {light_description} scene likely showing {best_label}, {motion_description}, between {start_sec}s and {end_sec}s."


def extract_video_chunks(
    video_path: Path,
    chunk_seconds: int,
    frames_per_chunk: int,
    processor: AutoProcessor,
    model: CLIPModel,
    device: torch.device,
) -> tuple[int, list[VideoChunk]]:
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
            summary_text = summarize_frames(frames, start_sec, end_sec, processor, model, device)
            chunks.append(
                VideoChunk(
                    chunk_index=chunk_index,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    frame_count=len(frames),
                    summary_text=summary_text,
                    embedding=embedding,
                )
            )
    finally:
        capture.release()

    return duration_sec, chunks


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = normalize(left)
    right_norm = normalize(right)
    return float(np.dot(left_norm, right_norm))


def ensure_embedding_dimension(chunks: list[VideoChunk], expected_dimension: int = 768) -> None:
    for chunk in chunks:
        actual_dimension = int(chunk.embedding.shape[0])
        if actual_dimension != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dimension}, got {actual_dimension}. "
                "Update infra/init.sql if you switch models."
            )


def load_clip_model() -> tuple[torch.device, AutoProcessor, CLIPModel]:
    device = get_device()
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_NAME)
    model = CLIPModel.from_pretrained(DEFAULT_MODEL_NAME).to(device)
    model.eval()
    return device, processor, model
