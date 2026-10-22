# llm-playground-corpus

llm-playground 프로젝트들이 함께 쓰는 코퍼스 모듈입니다. 패키지 이름은 `playground_corpus` 입니다.

- 코퍼스 타입: `Document` · `Section` · `Chunk`
- 토크나이저: `Tokenizer` Protocol, `HFTokenizer`(HF `tokenizers` 기반)
- 임베딩: `Embedder` Protocol
- Document ↔ parquet 읽기 · 쓰기 (`[parquet]` extras)

청커와 임베딩 모델 구현(`[local]` extras), 모델 비교 벤치는 이어서 추가합니다.

## 설치

```bash
# 기본 — 타입 · 토크나이저 · Protocol (numpy, tokenizers)
uv add playground-corpus

# parquet 읽기 · 쓰기
uv add "playground-corpus[parquet]"

# 임베딩 모델 실행 (torch, sentence-transformers)
uv add "playground-corpus[local]"
```

0.x 동안은 깨는 변경이 있을 수 있습니다. 첫 태그 전에는 같은 폴더에 나란히 clone 해서 path 의존으로 씁니다.

## 개발

```bash
# 전체 테스트 (extras 포함)
uv sync --all-extras
uv run pytest --ignore=tests/test_import_base.py
uv run ruff check

# base 설치에서 import 가 깨지지 않는지 (extras 없이)
uv sync
uv run pytest tests/test_import_base.py
```

## 구조

```
src/playground_corpus/   패키지
tests/                   단위 테스트
bench/                   모델 비교 벤치 (bench/data/ 는 git 제외)
reports/                 벤치 결과 — 집계 지표만 커밋
models/                  선정 결과 yaml
.ruff/                   공유 ruff 설정 복사본 (직접 고치지 않음)
```
