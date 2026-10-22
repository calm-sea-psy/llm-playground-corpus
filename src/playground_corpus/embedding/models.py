"""비교 후보 임베딩 모델 정의."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from playground_corpus.tokenizer import Tokenizer


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """임베딩 모델 하나의 고정 정보."""

    name: str
    """경로 · 컬렉션 이름에 쓰는 짧은 이름 (예: `bge-m3`)."""
    model_id: str
    """HF 허브 모델 id."""
    dim: int
    max_seq_length: int
    """모델이 받는 최대 토큰 수 (특수 토큰 포함)."""
    query_prefix: str = ""
    passage_prefix: str = ""

    def max_chunk_tokens(self, tokenizer: Tokenizer, special_token_count: int) -> int:
        """청크가 잘리지 않는 최대 토큰 수: 모델 최대 길이 − 특수 토큰 − 문서 접두어 토큰."""
        prefix_tokens = tokenizer.count(self.passage_prefix) if self.passage_prefix else 0
        return self.max_seq_length - special_token_count - prefix_tokens


# bge-m3 · KURE-v1 은 8192 까지 받지만 청크는 512 이하로 쓴다. KoE5 는 위치 임베딩 514 → 512
EMBEDDING_MODELS: Mapping[str, EmbeddingModelSpec] = MappingProxyType(
    {
        spec.name: spec
        for spec in (
            EmbeddingModelSpec(
                name="bge-m3", model_id="BAAI/bge-m3", dim=1024, max_seq_length=8192
            ),
            EmbeddingModelSpec(
                name="kure-v1", model_id="nlpai-lab/KURE-v1", dim=1024, max_seq_length=8192
            ),
            EmbeddingModelSpec(
                name="koe5",
                model_id="nlpai-lab/KoE5",
                dim=1024,
                max_seq_length=512,
                query_prefix="query: ",
                passage_prefix="passage: ",
            ),
        )
    }
)
