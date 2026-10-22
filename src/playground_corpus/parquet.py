"""Document 를 parquet 파일로 쓰고 읽는다. `[parquet]` extras 가 필요하다."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from types import ModuleType

from playground_corpus.types import Document, Section

BATCH_SIZE = 2_000  # 위키 문서 평균 수 KB 기준, 배치 하나가 수십 MB 를 넘지 않는 크기
_INSTALL_HINT = "parquet 읽기 · 쓰기에는 pyarrow 가 필요하다: uv add 'playground-corpus[parquet]'"


def write_documents(
    documents: Iterable[Document], path: Path, *, batch_size: int = BATCH_SIZE
) -> int:
    """문서를 parquet 파일 하나로 쓰고 쓴 문서 수를 돌려준다.

    Args:
      documents: 쓸 문서. 한 번에 메모리에 올리지 않고 batch_size 개씩 쓴다.
      path: 출력 파일. 상위 폴더가 없으면 만든다.
      batch_size: 한 번에 쓰는 문서 수. 1 이상.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size=}")
    pa, pq = _pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = _schema(pa)
    written = 0
    with pq.ParquetWriter(path, schema) as writer:
        batch: list[Document] = []
        for document in documents:
            batch.append(document)
            if len(batch) == batch_size:
                writer.write_table(_to_table(pa, schema, batch))
                written += len(batch)
                batch = []
        if batch:
            writer.write_table(_to_table(pa, schema, batch))
            written += len(batch)
    return written


def read_documents(path: Path) -> Iterator[Document]:
    """Parquet 파일의 문서를 쓴 순서대로 하나씩 돌려준다."""
    _, pq = _pyarrow()
    parquet_file = pq.ParquetFile(path)
    for record_batch in parquet_file.iter_batches():
        for row in record_batch.to_pylist():
            yield _from_row(row)


def _pyarrow() -> tuple[ModuleType, ModuleType]:
    try:
        import pyarrow as pa  # noqa: PLC0415  # [parquet] extras — base 설치에서 import 가 깨지지 않게
        import pyarrow.parquet as pq  # noqa: PLC0415  # [parquet] extras
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e
    return pa, pq


def _schema(pa: ModuleType) -> object:
    section = pa.struct([("level", pa.int16()), ("title", pa.string()), ("text", pa.string())])
    return pa.schema(
        [
            ("id", pa.string()),
            ("title", pa.string()),
            ("text", pa.string()),
            ("sections", pa.list_(section)),
            ("categories", pa.list_(pa.string())),
            ("meta", pa.map_(pa.string(), pa.string())),
        ]
    )


def _to_table(pa: ModuleType, schema: object, documents: list[Document]) -> object:
    rows = [
        {
            "id": document.id,
            "title": document.title,
            "text": document.text,
            "sections": [
                {"level": section.level, "title": section.title, "text": section.text}
                for section in document.sections
            ],
            "categories": list(document.categories),
            "meta": list(document.meta.items()),
        }
        for document in documents
    ]
    return pa.Table.from_pylist(rows, schema=schema)


def _from_row(row: dict[str, object]) -> Document:
    sections = tuple(
        Section(level=section["level"], title=section["title"], text=section["text"])
        for section in row["sections"]
    )
    return Document(
        id=row["id"],
        title=row["title"],
        text=row["text"],
        sections=sections,
        categories=tuple(row["categories"]),
        meta=dict(row["meta"]),
    )
