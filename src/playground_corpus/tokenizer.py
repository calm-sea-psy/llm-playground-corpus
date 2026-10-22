"""청커가 토큰 수를 셀 때 쓰는 토크나이저."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer as HFTokenizerBackend


class Tokenizer(Protocol):
    """임베딩 모델의 토크나이저. 청크 길이를 모델과 같은 기준으로 센다."""

    name: str
    """토크나이저 식별자. 모델 id 또는 `sha256:<tokenizer.json 해시 앞 16자>`."""

    def count(self, text: str) -> int:
        """특수 토큰을 빼고 센 토큰 수를 돌려준다."""
        ...

    def spans(self, text: str) -> list[tuple[int, int]]:
        """토큰마다 원문 글자 범위 (시작, 끝) 를 순서대로 돌려준다. 특수 토큰은 뺀다."""
        ...


@dataclass(frozen=True)
class _Encoded:
    ids: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]


class HFTokenizer:
    """HF `tokenizers` 로 감싼 토크나이저.

    예)
        tokenizer = HFTokenizer.from_pretrained("BAAI/bge-m3")
        tokenizer.count("서울은 대한민국의 수도이다.")
    """

    def __init__(self, backend: HFTokenizerBackend, *, name: str) -> None:
        self._backend = backend
        self.name = name

    @classmethod
    def from_pretrained(cls, model_id: str) -> HFTokenizer:
        """HF 허브에서 `tokenizer.json` 을 받아 만든다. 받은 파일은 HF 캐시에 남는다."""
        return cls(HFTokenizerBackend.from_pretrained(model_id), name=model_id)

    @classmethod
    def from_file(cls, path: Path) -> HFTokenizer:
        """로컬 `tokenizer.json` 으로 만든다. 식별자는 파일 내용 해시."""
        return cls(HFTokenizerBackend.from_file(str(path)), name=_content_name(path))

    @property
    def special_token_count(self) -> int:
        """한 입력에 붙는 특수 토큰 수 (예: `<s>` · `</s>`). 청크 최대 길이에서 이만큼 뺀다."""
        return self._backend.num_special_tokens_to_add(is_pair=False)

    def count(self, text: str) -> int:
        """특수 토큰을 빼고 센 토큰 수를 돌려준다."""
        return len(self._encode(text).ids)

    def spans(self, text: str) -> list[tuple[int, int]]:
        """토큰마다 원문 글자 범위 (시작, 끝) 를 순서대로 돌려준다. 특수 토큰은 뺀다."""
        return list(self._encode(text).spans)

    def _encode(self, text: str) -> _Encoded:
        encoding = self._backend.encode(text, add_special_tokens=False)
        return _Encoded(ids=tuple(encoding.ids), spans=tuple(encoding.offsets))


def _content_name(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest[:16]}"
