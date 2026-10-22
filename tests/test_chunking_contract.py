"""청커 · 실제 임베딩 모델 토크나이저 계약.

청커가 센 토큰 수가 모델이 실제로 쓰는 토크나이저(transformers)의 인코딩 길이와 같고,
특수 토큰을 붙여도 모델 최대 길이를 넘는 청크가 없어야 한다.
토크나이저 파일은 HF 허브에서 받아 캐시한다. 네트워크가 없으면 로컬에서는 skip, CI(`CI` 환경변수)에서는 실패.
"""

import os

import pytest

from playground_corpus import Chunker, ChunkingConfig, Document, HFTokenizer, Section

transformers = pytest.importorskip("transformers")

MODELS = ["BAAI/bge-m3", "nlpai-lab/KURE-v1", "nlpai-lab/KoE5"]
MAX_TOKENS = [256, 384, 512]

_PARAGRAPH = (
    "서울특별시는 대한민국의 수도이자 최대 도시이다. 조선 시대에는 한양(漢陽)으로 불렸으며, "
    "1394년 태조 이성계가 도읍을 옮긴 뒤 600년 넘게 수도 역할을 해 왔다. "
    "한강이 도시를 가로질러 흐르고, 북한산 · 관악산 등 산으로 둘러싸여 있다. "
    "2024년 기준 인구는 약 940만 명(9,386,034명)이며 수도권 인구는 2,600만 명을 넘는다. "
    "주요 산업은 금융 · IT · 관광이고, 1988년 하계 올림픽(Seoul Olympics)을 개최했다.\n"
)


def _load(model_id: str) -> tuple[HFTokenizer, object]:
    try:
        return (
            HFTokenizer.from_pretrained(model_id),
            transformers.AutoTokenizer.from_pretrained(model_id),
        )
    except Exception as e:  # 네트워크 · 허브 오류. CI 에서는 다시 던진다
        if os.environ.get("CI"):
            raise
        pytest.skip(f"{model_id} 토크나이저를 받을 수 없음: {type(e).__name__}")


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("max_tokens", MAX_TOKENS)
def test_chunk_token_count_matches_model_tokenizer(model_id: str, max_tokens: int):
    tokenizer, model_tokenizer = _load(model_id)
    document = Document(
        id="1",
        title="서울특별시",
        text=_PARAGRAPH * 32,
        sections=(Section(0, "", _PARAGRAPH * 2), Section(2, "역사", _PARAGRAPH * 30)),
    )

    chunks = Chunker(tokenizer, ChunkingConfig(max_tokens=max_tokens)).chunk(document)

    assert len(chunks) > 3  # 긴 섹션이 여러 창으로 나뉜 경우까지 확인
    long_section = document.sections[1].text.strip()
    assert (
        len(chunks) - 1 >= tokenizer.count(long_section) / max_tokens
    )  # 자르기 없이 끝까지 셌는지
    assert chunks[-1].text.endswith(long_section[-40:])  # 청크가 섹션 끝까지 덮는지
    for chunk in chunks:
        model_ids = model_tokenizer(chunk.text, add_special_tokens=False)["input_ids"]
        assert chunk.token_count == len(model_ids)
        assert chunk.token_count <= max_tokens
        assert chunk.token_count + tokenizer.special_token_count <= model_tokenizer.model_max_length
