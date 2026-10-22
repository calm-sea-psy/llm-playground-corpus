from pathlib import Path

import pytest

from playground_corpus import Document, Section, read_documents, write_documents

pytest.importorskip("pyarrow")


def _documents() -> list[Document]:
    return [
        Document(
            id="1",
            title="서울특별시",
            text="서울은 수도이다.\n\n조선의 수도였다.",
            sections=(Section(0, "", "서울은 수도이다."), Section(2, "역사", "조선의 수도였다.")),
            categories=("대한민국의 특별시",),
            meta={"source": "kowiki", "dump_date": "20261001"},
        ),
        Document(
            id="2", title="부산광역시", text="항구 도시.", sections=(Section(0, "", "항구 도시."),)
        ),
    ]


def test_write_then_read_returns_same_documents(tmp_path: Path):
    path = tmp_path / "parsed" / "part-0.parquet"

    written = write_documents(_documents(), path, batch_size=1)

    assert written == 2
    assert list(read_documents(path)) == _documents()


def test_write_rejects_zero_batch_size(tmp_path: Path):
    with pytest.raises(ValueError, match="batch_size"):
        write_documents(_documents(), tmp_path / "docs.parquet", batch_size=0)
