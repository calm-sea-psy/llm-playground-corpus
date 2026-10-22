from itertools import pairwise

import pytest

from playground_corpus import Chunker, ChunkingConfig, Document, HFTokenizer, Section


def _document(*sections: Section) -> Document:
    return Document(
        id="7",
        title="서울",
        text="\n\n".join(section.text for section in sections),
        sections=sections,
    )


def _sentences(count: int) -> str:
    return " ".join(f"문장 {i} 은 여기서 끝난다 ." for i in range(count))


def test_chunk_keeps_short_section_whole_with_header(word_tokenizer: HFTokenizer):
    body = "서울은 대한민국의 수도이다 . 조선 시대에는 한양으로 불렸다 ."
    chunker = Chunker(word_tokenizer, ChunkingConfig(max_tokens=50, min_chars=0))

    [lead, history] = chunker.chunk(_document(Section(0, "", body), Section(2, "역사", body)))

    assert lead.text == f"서울\n{body}"
    assert history.text == f"서울 > 역사\n{body}"
    assert (lead.id, lead.seq, history.id, history.seq) == ("7-0", 0, "7-1", 1)
    assert history.token_count == word_tokenizer.count(history.text)


def test_chunk_splits_long_section_within_max_tokens(word_tokenizer: HFTokenizer):
    config = ChunkingConfig(max_tokens=40, overlap_ratio=0.2, min_chars=0)
    body = _sentences(30)

    chunks = Chunker(word_tokenizer, config).chunk(_document(Section(2, "역사", body)))

    assert len(chunks) > 1
    assert all(chunk.token_count <= config.max_tokens for chunk in chunks)
    assert all(chunk.text.startswith("서울 > 역사\n") for chunk in chunks)
    assert "문장 0 " in chunks[0].text
    assert chunks[-1].text.endswith("문장 29 은 여기서 끝난다 .")


def test_chunk_overlaps_consecutive_windows(word_tokenizer: HFTokenizer):
    config = ChunkingConfig(max_tokens=40, overlap_ratio=0.2, min_chars=0)

    chunks = Chunker(word_tokenizer, config).chunk(_document(Section(2, "역사", _sentences(30))))

    for previous, current in pairwise(chunks):
        first_words = current.text.split("\n", 1)[1].split()[:3]
        assert " ".join(first_words) in previous.text


def test_chunk_ends_windows_at_sentence_boundaries(word_tokenizer: HFTokenizer):
    config = ChunkingConfig(max_tokens=40, overlap_ratio=0.0, min_chars=0)

    chunks = Chunker(word_tokenizer, config).chunk(_document(Section(2, "역사", _sentences(30))))

    assert all(chunk.text.endswith(".") for chunk in chunks)


def test_chunk_drops_bodies_shorter_than_min_chars(word_tokenizer: HFTokenizer):
    chunker = Chunker(word_tokenizer, ChunkingConfig(min_chars=10))

    chunks = chunker.chunk(
        _document(Section(0, "", "짧다 ."), Section(2, "생애", "충분히 긴 본문 문장이다 ."))
    )

    assert [chunk.section for chunk in chunks] == ["생애"]
    assert chunks[0].seq == 0


@pytest.mark.parametrize(
    "kwargs",
    [{"max_tokens": 0}, {"overlap_ratio": 0.5}, {"overlap_ratio": -0.1}, {"min_chars": -1}],
)
def test_chunking_config_rejects_invalid_values(kwargs: dict[str, float]):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        ChunkingConfig(**kwargs)
