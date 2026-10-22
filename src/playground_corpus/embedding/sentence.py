"""sentence-transformers 로 돌리는 임베더. `[local]` extras 가 필요하다."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from playground_corpus.embedding.models import EmbeddingModelSpec

if TYPE_CHECKING:  # [local] extras — base 설치에서 import 가 깨지지 않게 타입 검사 때만
    from sentence_transformers import SentenceTransformer

BATCH_SIZE = 64  # RTX 5080 16GB · fp16 · 512 토큰 기준 VRAM 여유를 두는 크기
_INSTALL_HINT = "임베딩 모델 실행에는 torch · sentence-transformers 가 필요하다: uv add 'playground-corpus[local]'"


class SentenceTransformerEmbedder:
    """sentence-transformers 모델로 질의 · 문서를 임베딩한다. 접두어는 모델 정의를 따른다.

    예)
        embedder = SentenceTransformerEmbedder.load(EMBEDDING_MODELS["bge-m3"])
        vectors = embedder.encode_passages([chunk.text for chunk in chunks])
    """

    def __init__(
        self, model: SentenceTransformer, spec: EmbeddingModelSpec, *, batch_size: int = BATCH_SIZE
    ) -> None:
        self._model = model
        self.spec = spec
        self._batch_size = batch_size

    @classmethod
    def load(
        cls,
        spec: EmbeddingModelSpec,
        *,
        device: str | None = None,
        fp16: bool = True,
        batch_size: int = BATCH_SIZE,
    ) -> SentenceTransformerEmbedder:
        """모델을 내려받아(캐시) 올린다. GPU 가 있으면 GPU, fp16 이면 반정밀도로."""
        try:
            import torch  # noqa: PLC0415  # [local] extras — base 설치에서 import 가 깨지지 않게
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415  # [local] extras
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer(spec.model_id, device=device)
        model.max_seq_length = spec.max_seq_length
        if fp16 and device.startswith("cuda"):
            model.half()
        return cls(model, spec, batch_size=batch_size)

    def encode_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """검색 질의 n개를 (n, d) 로 임베딩한다. 행마다 L2 정규화."""
        return self._encode([self.spec.query_prefix + text for text in texts])

    def encode_passages(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """문서 청크 n개를 (n, d) 로 임베딩한다. 행마다 L2 정규화."""
        return self._encode([self.spec.passage_prefix + text for text in texts])

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        # sentence-transformers 가 길이순으로 정렬해 배치를 만들고 입력 순서로 되돌려 준다
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
