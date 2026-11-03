import pickle
from dataclasses import FrozenInstanceError

import pytest

from playground_corpus import Chunk, Document, Section


def test_document_defaults_to_empty_categories_and_meta():
    doc = Document(id="1", title="서울특별시", text="본문", sections=(Section(0, "", "본문"),))

    assert doc.categories == ()
    assert doc.meta == {}


def test_document_survives_pickle_between_processes():
    doc = Document(
        id="1",
        title="서울특별시",
        text="본문",
        sections=(Section(0, "", "본문"),),
        categories=("도시",),
        meta={"source": "kowiki"},
    )

    assert pickle.loads(pickle.dumps(doc)) == doc  # noqa: S301  # 방금 만든 바이트


def test_chunk_is_frozen():
    chunk = Chunk(
        id="1-0", doc_id="1", title="서울특별시", section="역사", text="본문", seq=0, token_count=2
    )

    with pytest.raises(FrozenInstanceError):
        chunk.seq = 1  # type: ignore[misc]  # frozen 확인용
