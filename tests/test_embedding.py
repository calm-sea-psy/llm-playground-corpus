import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from playground_corpus import EMBEDDING_MODELS, EmbeddingCache, EmbeddingModelSpec, HFTokenizer

_SPEC = EmbeddingModelSpec(
    name="fake",
    model_id="test/fake",
    dim=4,
    max_seq_length=16,
    query_prefix="q: ",
    passage_prefix="p: ",
)


class FakeEmbedder:
    """텍스트 해시로 정해지는 벡터. 받은 텍스트를 기록한다."""

    spec = _SPEC

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._encode([self.spec.query_prefix + text for text in texts])

    def encode_passages(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._encode([self.spec.passage_prefix + text for text in texts])

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        self.calls.append(texts)
        rows = [
            np.frombuffer(hashlib.sha256(t.encode()).digest()[:16], dtype=np.uint8) for t in texts
        ]
        vectors = np.asarray(rows, dtype=np.float32).reshape(len(texts), -1)[:, : self.spec.dim]
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_cache_writes_shards_and_loads_in_order(tmp_path: Path):
    ids = [f"d-{i}" for i in range(7)]
    texts = [f"청크 {i}" for i in range(7)]
    embedder = FakeEmbedder()

    info = EmbeddingCache(tmp_path, shard_size=3).write(embedder, ids, texts)
    loaded_ids, vectors = EmbeddingCache(tmp_path).load()

    assert info.shards == ("emb-00000", "emb-00001", "emb-00002")
    assert (info.model, info.kind, info.dim, info.count) == ("fake", "passage", 4, 7)
    assert loaded_ids == ids
    assert vectors.shape == (7, 4)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(vectors, FakeEmbedder().encode_passages(texts), rtol=1e-6)


def test_cache_skips_shards_already_written(tmp_path: Path):
    ids = [f"d-{i}" for i in range(7)]
    texts = [f"청크 {i}" for i in range(7)]
    EmbeddingCache(tmp_path, shard_size=3).write(FakeEmbedder(), ids, texts)
    (tmp_path / "emb-00002.npy").unlink()  # 마지막 샤드만 지워 중단된 상황을 만든다

    embedder = FakeEmbedder()
    EmbeddingCache(tmp_path, shard_size=3).write(embedder, ids, texts)

    assert embedder.calls == [["p: 청크 6"]]


def test_cache_uses_query_prefix_for_queries(tmp_path: Path):
    embedder = FakeEmbedder()

    info = EmbeddingCache(tmp_path).write(embedder, ["q-1"], ["서울은?"], kind="query")

    assert info.kind == "query"
    assert embedder.calls == [["q: 서울은?"]]


def test_cache_rejects_mismatched_ids_and_texts(tmp_path: Path):
    with pytest.raises(ValueError, match="same length"):
        EmbeddingCache(tmp_path).write(FakeEmbedder(), ["a"], ["x", "y"])


def test_max_chunk_tokens_leaves_room_for_specials_and_prefix(word_tokenizer: HFTokenizer):
    spec = EmbeddingModelSpec(
        name="e5", model_id="x", dim=4, max_seq_length=512, passage_prefix="passage: here"
    )

    # 접두어는 단어 단위로 "passage" · ":" · "here" 3개
    assert spec.max_chunk_tokens(word_tokenizer, special_token_count=2) == 512 - 2 - 3


def test_embedding_models_define_three_candidates():
    assert set(EMBEDDING_MODELS) == {"bge-m3", "kure-v1", "koe5"}
    assert EMBEDDING_MODELS["koe5"].passage_prefix == "passage: "
    assert EMBEDDING_MODELS["koe5"].max_seq_length == 512
