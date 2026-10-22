"""문서 · 섹션 · 청크 자료형."""

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    """문서 안의 섹션 하나."""

    level: int
    """헤더 깊이. 문서 첫머리(첫 헤더 앞 본문)는 0. 위키 `== 제목 ==` 은 2."""
    title: str
    """섹션 제목. 문서 첫머리는 빈 문자열."""
    text: str
    """정제된 섹션 본문."""


@dataclass(frozen=True)
class Document:
    """코퍼스가 내놓는 정제된 문서 하나."""

    id: str
    """코퍼스 안에서 유일한 문서 id. 위키는 페이지 id."""
    title: str
    text: str
    """정제된 전체 본문. 섹션 본문을 순서대로 이은 것."""
    sections: tuple[Section, ...]
    categories: tuple[str, ...] = ()
    """분류 이름. 본문에서 빼서 여기에 둔다."""
    meta: Mapping[str, str] = field(default_factory=dict)
    """source · url · dump_date 등 문자열 메타데이터."""


@dataclass(frozen=True)
class Chunk:
    """임베딩 · 검색 단위가 되는 문서 조각."""

    id: str
    """`{doc_id}-{seq}` 형태의 청크 id."""
    doc_id: str
    title: str
    """원문서 제목."""
    section: str
    """원문서 섹션 제목. 문서 첫머리면 빈 문자열."""
    text: str
    """`"{제목} > {섹션}\\n{본문}"` 형태. 이 문자열을 그대로 임베딩한다."""
    seq: int
    """문서 안 청크 순서. 0부터."""
