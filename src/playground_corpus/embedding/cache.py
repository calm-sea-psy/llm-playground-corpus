"""임베딩 결과를 npy 샤드로 저장하고 읽는 캐시. 한 번 계산한 벡터는 다시 계산하지 않는다."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from playground_corpus.embedding.protocol import Embedder

logger = logging.getLogger(__name__)

SHARD_SIZE = 50_000  # 1024d float32 기준 샤드 하나 약 200MB
MANIFEST = "manifest.json"
DTYPE = np.float32

Kind = Literal["passage", "query"]


@dataclass(frozen=True)
class EmbeddingCacheInfo:
    """캐시 폴더 하나의 내용."""

    model: str
    """임베딩 모델 짧은 이름."""
    model_id: str
    kind: Kind
    dim: int
    count: int
    shards: tuple[str, ...]
    """샤드 이름 (`emb-00000`). 각각 `{이름}.npy` 벡터와 `{이름}.ids.json` id 목록."""


class EmbeddingCache:
    """한 모델 · 한 종류(문서/질의)의 벡터를 `emb-xxxxx.npy` + `emb-xxxxx.ids.json` 샤드로 둔다.

    이미 있는 샤드는 건너뛰므로 중단된 작업을 이어서 돌릴 수 있다.

    예)
        cache = EmbeddingCache(Path("data/emb/bge-m3/kowiki-subset-10000-384"))
        info = cache.write(embedder, ids, texts)
        ids, vectors = cache.load()
    """

    def __init__(self, directory: Path, *, shard_size: int = SHARD_SIZE) -> None:
        if shard_size < 1:
            raise ValueError(f"shard_size must be >= 1, got {shard_size=}")
        self.directory = directory
        self._shard_size = shard_size

    def write(
        self,
        embedder: Embedder,
        ids: Sequence[str],
        texts: Sequence[str],
        *,
        kind: Kind = "passage",
    ) -> EmbeddingCacheInfo:
        """텍스트를 샤드 단위로 임베딩해 저장하고 캐시 정보를 돌려준다. 있는 샤드는 다시 계산하지 않는다.

        Args:
          embedder: 임베딩 모델.
          ids: 텍스트마다 붙는 id. texts 와 길이가 같아야 한다.
          texts: 임베딩할 텍스트.
          kind: `passage` 면 문서 접두어, `query` 면 질의 접두어를 쓴다.
        """
        if len(ids) != len(texts):
            raise ValueError(
                f"ids and texts must have the same length, got {len(ids)=} {len(texts)=}"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        encode = embedder.encode_passages if kind == "passage" else embedder.encode_queries
        shards: list[str] = []
        for number, start in enumerate(range(0, len(texts), self._shard_size)):
            name = f"emb-{number:05d}"
            shards.append(name)
            shard_ids = list(ids[start : start + self._shard_size])
            if self._shard_done(name, shard_ids):
                logger.info("skip %s (already embedded)", name)
                continue
            vectors = encode(texts[start : start + self._shard_size]).astype(DTYPE)
            np.save(self.directory / f"{name}.npy", vectors)
            (self.directory / f"{name}.ids.json").write_text(
                json.dumps(shard_ids, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("wrote %s (%d vectors)", name, len(shard_ids))
        info = EmbeddingCacheInfo(
            model=embedder.spec.name,
            model_id=embedder.spec.model_id,
            kind=kind,
            dim=embedder.spec.dim,
            count=len(ids),
            shards=tuple(shards),
        )
        (self.directory / MANIFEST).write_text(
            json.dumps(asdict(info), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return info

    def load(self, *, mmap: bool = False) -> tuple[list[str], NDArray[np.float32]]:
        """저장된 id 와 벡터를 쓴 순서대로 돌려준다.

        Args:
          mmap: True 면 샤드를 메모리 매핑으로 연다 (전체 규모에서 메모리를 아낀다). 그래도 이어 붙이면 복사된다.

        Returns:
          (ids, (n, d) 벡터).
        """
        info = self.info()
        ids: list[str] = []
        parts: list[NDArray[np.float32]] = []
        for name in info.shards:
            ids.extend(
                json.loads((self.directory / f"{name}.ids.json").read_text(encoding="utf-8"))
            )
            parts.append(np.load(self.directory / f"{name}.npy", mmap_mode="r" if mmap else None))
        vectors = np.concatenate(parts) if parts else np.empty((0, info.dim), dtype=DTYPE)
        return ids, vectors

    def info(self) -> EmbeddingCacheInfo:
        """manifest.json 을 읽는다."""
        raw = json.loads((self.directory / MANIFEST).read_text(encoding="utf-8"))
        return EmbeddingCacheInfo(**{**raw, "shards": tuple(raw["shards"])})

    def _shard_done(self, name: str, shard_ids: list[str]) -> bool:
        ids_path = self.directory / f"{name}.ids.json"
        if not ids_path.exists() or not (self.directory / f"{name}.npy").exists():
            return False
        return json.loads(ids_path.read_text(encoding="utf-8")) == shard_ids
