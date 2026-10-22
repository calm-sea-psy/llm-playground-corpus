"""임베딩 계약 · 후보 모델 정의 · sentence-transformers 구현 · 벡터 캐시."""

from playground_corpus.embedding.cache import EmbeddingCache, EmbeddingCacheInfo
from playground_corpus.embedding.models import EMBEDDING_MODELS, EmbeddingModelSpec
from playground_corpus.embedding.protocol import Embedder
from playground_corpus.embedding.sentence import SentenceTransformerEmbedder

__all__ = [
    "EMBEDDING_MODELS",
    "Embedder",
    "EmbeddingCache",
    "EmbeddingCacheInfo",
    "EmbeddingModelSpec",
    "SentenceTransformerEmbedder",
]
