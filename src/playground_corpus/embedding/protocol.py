"""텍스트를 벡터로 바꾸는 임베딩 계약."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from playground_corpus.embedding.models import EmbeddingModelSpec


class Embedder(Protocol):
    """텍스트를 벡터로 바꾸는 임베딩 모델. 질의와 문서에 붙는 접두어가 모델마다 달라 둘을 나눈다."""

    spec: EmbeddingModelSpec

    def encode_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """검색 질의 n개를 임베딩한다.

        Args:
          texts: 질의 n개. 모델의 질의 접두어는 구현이 붙인다.

        Returns:
          (n, d). 행마다 L2 정규화돼 있다.
        """
        ...

    def encode_passages(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """문서 청크 n개를 임베딩한다.

        Args:
          texts: 청크 텍스트 n개. 모델의 문서 접두어는 구현이 붙인다.

        Returns:
          (n, d). 행마다 L2 정규화돼 있다.
        """
        ...
