"""문서를 섹션 단위로 나누고, 긴 섹션은 토큰 기준 창으로 다시 나누는 청커."""

import re
from dataclasses import dataclass

from playground_corpus.tokenizer import Tokenizer
from playground_corpus.types import Chunk, Document, Section

HEADER_SEPARATOR = " > "
MIN_WINDOW_RATIO = 0.6  # 경계를 찾을 때 창을 이 비율 아래로는 줄이지 않는다 (너무 짧은 청크 방지)

# 창 끝을 맞출 자연스러운 경계 (폭 0 위치). 문단 > 문장 순으로 찾는다
_PARAGRAPH_END = re.compile(r"(?=\n)")
_SENTENCE_END = re.compile(r"(?<=[.!?。])(?=\s)")


@dataclass(frozen=True)
class ChunkingConfig:
    """청킹 설정."""

    max_tokens: int = 384
    """청크 한 개(헤더 포함)의 최대 토큰 수. 특수 토큰은 뺀 수."""
    overlap_ratio: float = 0.15
    """긴 섹션을 나눌 때 앞 청크와 겹치는 본문 토큰 비율. 0 이상 0.5 미만."""
    min_chars: int = 50
    """본문(헤더 제외)이 이보다 짧은 청크는 버린다."""

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens=}")
        if not 0 <= self.overlap_ratio < 0.5:  # noqa: PLR2004  # 절반 이상 겹치면 창이 거의 나아가지 않는다
            raise ValueError(f"overlap_ratio must be in [0, 0.5), got {self.overlap_ratio=}")
        if self.min_chars < 0:
            raise ValueError(f"min_chars must be >= 0, got {self.min_chars=}")


@dataclass(frozen=True)
class _Window:
    start_char: int
    end_char: int
    end_token: int


class Chunker:
    """섹션 단위로 먼저 나누고, 헤더 포함 `max_tokens` 를 넘는 섹션은 오버랩을 둔 창으로 나눈다.

    청크 길이는 임베딩 모델과 같은 토크나이저로 센다. 창 끝은 가능하면 문단 · 문장 경계에 맞춘다.

    예)
        chunker = Chunker(HFTokenizer.from_pretrained("BAAI/bge-m3"), ChunkingConfig(max_tokens=384))
        chunks = chunker.chunk(document)
    """

    def __init__(self, tokenizer: Tokenizer, config: ChunkingConfig | None = None) -> None:
        self._tokenizer = tokenizer
        self.config = config or ChunkingConfig()

    def chunk(self, document: Document) -> list[Chunk]:
        """문서를 청크 목록으로 바꾼다. 순서는 섹션 순서, `seq` 는 0부터."""
        chunks: list[Chunk] = []
        for section in document.sections:
            header = _header(document.title, section)
            for body in self._split_section(header, section.text):
                if len(body) < self.config.min_chars:
                    continue
                text = f"{header}\n{body}"
                seq = len(chunks)
                chunks.append(
                    Chunk(
                        id=f"{document.id}-{seq}",
                        doc_id=document.id,
                        title=document.title,
                        section=section.title,
                        text=text,
                        seq=seq,
                        token_count=self._tokenizer.count(text),
                    )
                )
        return chunks

    def _split_section(self, header: str, body: str) -> list[str]:
        """헤더를 붙였을 때 max_tokens 안에 드는 본문 조각들을 돌려준다."""
        body = body.strip()
        if self._fits(header, body):
            return [body]
        spans = self._tokenizer.spans(body)
        budget = max(1, self.config.max_tokens - self._tokenizer.count(f"{header}\n"))
        overlap = int(budget * self.config.overlap_ratio)
        pieces: list[str] = []
        start_token = 0
        while start_token < len(spans):
            window = self._window(header, body, spans, start_token, budget)
            pieces.append(body[window.start_char : window.end_char].strip())
            if window.end_token >= len(spans):
                break
            start_token = max(start_token + 1, window.end_token - overlap)
        return [piece for piece in pieces if piece]

    def _window(
        self, header: str, body: str, spans: list[tuple[int, int]], start_token: int, budget: int
    ) -> _Window:
        """start_token 부터 budget 토큰 안에서 경계에 맞춘 창. 합친 텍스트가 max_tokens 를 넘으면 줄인다."""
        end_token = min(len(spans), start_token + budget)
        if end_token < len(spans):
            end_token = _snap_to_boundary(body, spans, start_token, end_token)
        start_char = spans[start_token][0]
        while True:
            end_char = spans[end_token - 1][1]
            if end_token - start_token <= 1 or self._fits(
                header, body[start_char:end_char].strip()
            ):
                return _Window(start_char, end_char, end_token)
            # 헤더와 붙이면서 경계 토큰이 달라져 넘친 경우 — 넘친 만큼 줄여 다시 센다
            overflow = (
                self._tokenizer.count(f"{header}\n{body[start_char:end_char].strip()}")
                - self.config.max_tokens
            )
            end_token = max(start_token + 1, end_token - max(1, overflow))

    def _fits(self, header: str, body: str) -> bool:
        return self._tokenizer.count(f"{header}\n{body}") <= self.config.max_tokens


def _header(title: str, section: Section) -> str:
    return f"{title}{HEADER_SEPARATOR}{section.title}" if section.title else title


def _snap_to_boundary(
    body: str, spans: list[tuple[int, int]], start_token: int, end_token: int
) -> int:
    """창 끝을 문단 → 문장 경계로 당긴다. 창이 MIN_WINDOW_RATIO 아래로 줄면 그대로 둔다."""
    min_end = start_token + max(1, int((end_token - start_token) * MIN_WINDOW_RATIO))
    window_text_start = spans[min_end - 1][1]
    window_text_end = spans[end_token - 1][1]
    for pattern in (_PARAGRAPH_END, _SENTENCE_END):
        boundary = None
        for match in pattern.finditer(body, window_text_start, window_text_end):
            boundary = match.start()
        if boundary is not None:
            # boundary 문자 위치 이전에 끝나는 마지막 토큰까지
            for token in range(end_token, min_end - 1, -1):
                if spans[token - 1][1] <= boundary:
                    return token
    return end_token
