from pathlib import Path

import pytest
from tokenizers import Tokenizer as Backend
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from playground_corpus import HFTokenizer


@pytest.fixture
def tokenizer_file(tmp_path: Path) -> Path:
    # 네트워크 없이 쓰는 작은 단어 단위 토크나이저
    vocab = {"[UNK]": 0, "서울은": 1, "수도이다": 2, ".": 3}
    backend = Backend(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))
    return path


def test_count_counts_tokens_without_special_tokens(tokenizer_file: Path):
    tokenizer = HFTokenizer.from_file(tokenizer_file)

    assert tokenizer.count("서울은 수도이다.") == 3


def test_spans_point_to_original_text(tokenizer_file: Path):
    text = "서울은 수도이다."
    tokenizer = HFTokenizer.from_file(tokenizer_file)

    pieces = [text[start:end] for start, end in tokenizer.spans(text)]

    assert pieces == ["서울은", "수도이다", "."]


def test_count_ignores_truncation_in_tokenizer_file(tmp_path: Path):
    backend = Backend(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.enable_truncation(max_length=4)  # KoE5 처럼 tokenizer.json 에 자르기가 들어 있는 경우
    path = tmp_path / "truncating.json"
    backend.save(str(path))

    tokenizer = HFTokenizer.from_file(path)

    assert tokenizer.count("하나 둘 셋 넷 다섯 여섯") == 6


def test_from_file_names_tokenizer_by_content_hash(tokenizer_file: Path):
    tokenizer = HFTokenizer.from_file(tokenizer_file)

    assert tokenizer.name.startswith("sha256:")
    assert len(tokenizer.name) == len("sha256:") + 16
