from pathlib import Path

import pytest
from tokenizers import Tokenizer as Backend
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from playground_corpus import HFTokenizer


@pytest.fixture
def word_tokenizer(tmp_path: Path) -> HFTokenizer:
    """네트워크 없이 쓰는 단어 단위 토크나이저. 모든 단어가 [UNK] 지만 개수 · 위치는 정확하다."""
    backend = Backend(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    path = tmp_path / "word-tokenizer.json"
    backend.save(str(path))
    return HFTokenizer.from_file(path)
