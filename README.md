# llm-playground-corpus

llm-playground 프로젝트들이 함께 쓰는 코퍼스 모듈입니다. 패키지 이름은 `playground_corpus` 입니다.

- 코퍼스 타입: `Document` · `Section` · `Chunk`
- 토크나이저: `Tokenizer` Protocol, `HFTokenizer`(HF `tokenizers` 기반)
- 청커: `Chunker` · `ChunkingConfig` — 섹션 단위, 긴 섹션은 오버랩을 둔 토큰 창
- 임베딩: `Embedder` Protocol, 후보 모델 `EMBEDDING_MODELS`(bge-m3 · KURE-v1 · KoE5), `SentenceTransformerEmbedder`(`[local]` extras), npy 샤드 캐시 `EmbeddingCache`
- Document ↔ parquet 읽기 · 쓰기 (`[parquet]` extras)

## 벤치 — 임베딩 모델 × 청크 크기 비교

llm-playground-wiki(옛 llm-playground-rag)가 만든 서브셋과 KorQuAD dev 로 검색 품질(hit@k · MRR) · 처리량 · VRAM 을 잽니다.
서브셋 데이터를 읽으므로 llm-playground-wiki 를 같은 부모 폴더에 나란히 clone 해 두어야 합니다. evalkit 은 git 태그로 설치됩니다.

```bash
uv run --group bench --extra parquet --extra local python bench/embed_compare.py
# 시험: 모델 · 크기 · 문서 수를 줄여서
uv run --group bench --extra parquet --extra local python bench/embed_compare.py --models bge-m3 --max-tokens 384 --limit-docs 300 --run-id smoke
```

임베딩은 `bench/data/emb/` 에 캐시해 다시 계산하지 않고, 집계 표는 `reports/{run_id}.md` 에 씁니다.

## 설치

```bash
# 기본 — 타입 · 토크나이저 · Protocol (numpy, tokenizers)
uv add "playground-corpus @ git+https://github.com/calm-sea-psy/llm-playground-corpus@v0.1.0"

# parquet 읽기 · 쓰기
uv add "playground-corpus[parquet] @ git+https://github.com/calm-sea-psy/llm-playground-corpus@v0.1.0"

# 임베딩 모델 실행 (torch, sentence-transformers)
uv add "playground-corpus[local] @ git+https://github.com/calm-sea-psy/llm-playground-corpus@v0.1.0"
```

0.x 동안은 깨는 변경이 있을 수 있습니다. 이 레포와 함께 고칠 때는 같은 부모 폴더에 나란히 clone 하고 잠깐 path 로 바꿔 씁니다: `{ path = "../llm-playground-corpus", editable = true }`.

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
