"""텍스트를 벡터로 바꾸는 임베딩 계약."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = ["Embedder"]


class Embedder(Protocol):
    """텍스트를 벡터로 바꾸는 임베딩 모델."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """텍스트 묶음을 임베딩한다.

        Args:
          texts: 임베딩할 텍스트 n개.

        Returns:
          (n, d). 행마다 L2 정규화돼 있다.
        """
        ...
