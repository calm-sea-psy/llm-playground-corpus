from dataclasses import FrozenInstanceError

import pytest

from playground_corpus import Chunk, Document, Section


def test_document_defaults_to_empty_categories_and_meta():
    doc = Document(id="1", title="서울특별시", text="본문", sections=(Section(0, "", "본문"),))

    assert doc.categories == ()
    assert doc.meta == {}


def test_chunk_is_frozen():
    chunk = Chunk(id="1-0", doc_id="1", title="서울특별시", section="역사", text="본문", seq=0)

    with pytest.raises(FrozenInstanceError):
        chunk.seq = 1  # type: ignore[misc]  # frozen 확인용
