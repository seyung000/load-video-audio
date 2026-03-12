import numpy as np

# 벡터 정규화 및 pgvector에 맞는 문자열 변환 유틸리티
def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm

# pgvector에 벡터를 삽입할 때 사용할 문자열 포맷으로 변환
def vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vector.tolist()) + "]"