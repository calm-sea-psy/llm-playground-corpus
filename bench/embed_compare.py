"""임베딩 모델 × 청크 크기 비교 벤치: KorQuAD dev 로 검색 품질 · 처리량 · VRAM 을 잰다.

입력은 llm-playground-rag 가 만든 서브셋(parquet + subset.json)과 KorQuAD dev 다.
이 레포는 RAG 를 import 하지 않고 파일로만 받는다.

검색은 numpy 전수 비교(정확한 최근접)로 한다. 서브셋 수치는 모델 · 설정 간 상대 비교 전용이다.

출력:
  bench/data/emb/{모델}/...        임베딩 캐시 (다시 계산하지 않음, git 제외)
  bench/data/runs/{run_id}/        문항별 기록 jsonl · summary.json (git 제외)
  reports/{run_id}.md              집계 표 (커밋)

사용법:
  uv run --group bench --extra parquet --extra local python bench/embed_compare.py
  uv run --group bench --extra parquet --extra local python bench/embed_compare.py \
      --models bge-m3 --max-tokens 384 --limit-docs 300 --run-id smoke
"""

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from playground_evalkit import (
    KorQuADItem,
    ResultRecord,
    contains_answer,
    hit_at_k,
    load_korquad,
    mrr,
    render_table,
    write_records,
)

from playground_corpus import (
    EMBEDDING_MODELS,
    Chunk,
    Chunker,
    ChunkingConfig,
    Document,
    EmbeddingCache,
    HFTokenizer,
    SentenceTransformerEmbedder,
    read_documents,
)

DEFAULT_SUBSET = Path("../llm-playground-rag/data/parsed/kowiki-subset-10000")
OUT_DIR = Path("bench/data")
REPORTS_DIR = Path("reports")
MAX_TOKENS = (256, 384, 512)
TOP_K = 10  # hit@10 · MRR@10 까지 본다
QUERY_BATCH = (
    512  # 질의 × 청크 점수 행렬을 나눠 계산하는 크기 (5766 × 6.5만 float32 를 한 번에 올리지 않게)
)


@dataclass(frozen=True)
class RunResult:
    """모델 × 청크 크기 한 조합의 집계."""

    model: str
    max_tokens: int
    """요청한 청크 크기."""
    effective_max_tokens: int
    """모델 한도(특수 토큰 · 접두어 몫)를 반영해 실제 쓴 크기."""
    chunks: int
    queries: int
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    mrr_at_10: float
    doc_hit_at_5: float
    """상위 5개 청크에 정답 문서(제목)가 있는 비율 — 정답 문자열 포함 여부는 보지 않는다."""
    chars_per_token: float
    tokenizer: str
    passages_per_sec: float | None
    """이번 실행에서 새로 임베딩했을 때만. 캐시에서 읽었으면 None."""
    vram_peak_gb: float | None


def main() -> None:
    args = _parse_args()
    run_id = args.run_id or f"{datetime.now(tz=UTC):%Y-%m-%d}-embed-compare"
    subset_dir: Path = args.subset
    summary = json.loads((subset_dir / "subset.json").read_text(encoding="utf-8"))
    documents = list(read_documents(subset_dir / "docs-00000.parquet"))
    korquad = args.korquad or subset_dir.parents[2] / summary["korquad_dev"].replace("\\", "/")
    items = load_korquad(korquad, title_aliases=summary["korquad_title_map"])
    documents, items = _limit(documents, items, args.limit_docs)
    print(f"documents={len(documents)} questions={len(items)} run_id={run_id}")

    results: list[RunResult] = []
    run_dir = OUT_DIR / "runs" / run_id
    for model_name in args.models:
        spec = EMBEDDING_MODELS[model_name]
        tokenizer = HFTokenizer.from_pretrained(spec.model_id)
        embedder = SentenceTransformerEmbedder.load(spec, batch_size=args.batch_size)
        query_cache = EmbeddingCache(
            OUT_DIR / "emb" / spec.name / f"queries-{subset_dir.name}-{len(documents)}"
        )
        query_cache.write(
            embedder, [item.id for item in items], [item.question for item in items], kind="query"
        )
        _, queries = query_cache.load()
        limit = spec.max_chunk_tokens(tokenizer, tokenizer.special_token_count)
        for max_tokens in args.max_tokens:
            effective = min(max_tokens, limit)
            chunks = _chunk(documents, Chunker(tokenizer, ChunkingConfig(max_tokens=effective)))
            cache = EmbeddingCache(
                OUT_DIR / "emb" / spec.name / f"{subset_dir.name}-{len(documents)}-{effective}"
            )
            speed, vram = _embed(cache, embedder, chunks)
            _, passages = cache.load()
            top = _top_k(queries, passages)
            result, records = _score(
                run_id=run_id,
                model=spec.name,
                max_tokens=max_tokens,
                effective=effective,
                items=items,
                chunks=chunks,
                top=top,
                tokenizer_name=tokenizer.name,
            )
            result = RunResult(
                **{**asdict(result), "passages_per_sec": speed, "vram_peak_gb": vram}
            )
            write_records(run_dir / f"records-{spec.name}-{max_tokens}.jsonl", records)
            results.append(result)
            print(
                f"{spec.name:8s} {max_tokens}(→{effective}) chunks={result.chunks:,} hit@5={result.hit_at_5:.3f} "
                f"mrr={result.mrr_at_10:.3f} doc_hit@5={result.doc_hit_at_5:.3f} speed={speed}"
            )
        del embedder
        torch.cuda.empty_cache()

    _write_outputs(run_id, run_dir, results, documents, items, args)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument(
        "--korquad", type=Path, default=None, help="기본: subset.json 의 korquad_dev"
    )
    parser.add_argument(
        "--models", nargs="+", default=list(EMBEDDING_MODELS), choices=list(EMBEDDING_MODELS)
    )
    parser.add_argument("--max-tokens", nargs="+", type=int, default=list(MAX_TOKENS))
    parser.add_argument(
        "--limit-docs", type=int, default=0, help="앞에서부터 이만큼 문서만 (0 = 전체, 시험용)"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def _limit(
    documents: list[Document], items: list[KorQuADItem], limit: int
) -> tuple[list[Document], list[KorQuADItem]]:
    """문서를 줄이면 질문도 남은 문서의 것만. 질문이 있는 문서를 먼저 남긴다."""
    titles = {item.title for item in items}
    if limit:
        with_questions = [doc for doc in documents if doc.title in titles]
        others = [doc for doc in documents if doc.title not in titles]
        documents = (with_questions + others)[:limit]
    kept = {doc.title for doc in documents}
    return documents, [item for item in items if item.title in kept]


def _chunk(documents: list[Document], chunker: Chunker) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunker.chunk(document)]


def _embed(
    cache: EmbeddingCache, embedder: SentenceTransformerEmbedder, chunks: list[Chunk]
) -> tuple[float | None, float | None]:
    """캐시에 없는 샤드만 임베딩하고, 새로 계산했으면 (청크/초, VRAM 최대 GB) 를 돌려준다."""
    ids = [chunk.id for chunk in chunks]
    cached = (cache.directory / "manifest.json").exists() and cache.load()[0] == ids
    if cached:
        return None, None
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    cache.write(embedder, ids, [chunk.text for chunk in chunks])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return round(len(chunks) / elapsed, 1), round(torch.cuda.max_memory_allocated() / 2**30, 2)


def _top_k(queries: NDArray[np.float32], passages: NDArray[np.float32]) -> NDArray[np.int64]:
    """질의마다 코사인 유사도 상위 TOP_K 청크 인덱스 (높은 순).

    Args:
      queries: (q, d). L2 정규화돼 있다.
      passages: (n, d). L2 정규화돼 있다.

    Returns:
      (q, TOP_K).
    """
    k = min(TOP_K, len(passages))
    rows = []
    for start in range(0, len(queries), QUERY_BATCH):
        scores = queries[start : start + QUERY_BATCH] @ passages.T
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
        rows.append(np.take_along_axis(part, order, axis=1))
    return np.concatenate(rows)


def _score(
    *,
    run_id: str,
    model: str,
    max_tokens: int,
    effective: int,
    items: list[KorQuADItem],
    chunks: list[Chunk],
    top: NDArray[np.int64],
    tokenizer_name: str,
) -> tuple[RunResult, list[ResultRecord]]:
    relevance = [
        [contains_answer(chunks[i].title, chunks[i].text, item) for i in row]
        for item, row in zip(items, top, strict=True)
    ]
    doc_relevance = [
        [chunks[i].title == item.title for i in row] for item, row in zip(items, top, strict=True)
    ]
    created_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    spec = {
        "model": model,
        "max_tokens": max_tokens,
        "effective_max_tokens": effective,
        "tokenizer": tokenizer_name,
    }
    records = [
        ResultRecord(
            run_id=run_id,
            spec_id=f"{model}/{max_tokens}",
            item_id=item.id,
            scores={
                "hit@1": float(any(rel[:1])),
                "hit@5": float(any(rel[:5])),
                "hit@10": float(any(rel[:10])),
                "rr@10": next((1.0 / rank for rank, hit in enumerate(rel, start=1) if hit), 0.0),
                "doc_hit@5": float(any(doc[:5])),
            },
            spec=spec,
            created_at=created_at,
        )
        for item, rel, doc in zip(items, relevance, doc_relevance, strict=True)
    ]
    chars = sum(len(chunk.text) for chunk in chunks)
    tokens = sum(chunk.token_count for chunk in chunks)
    result = RunResult(
        model=model,
        max_tokens=max_tokens,
        effective_max_tokens=effective,
        chunks=len(chunks),
        queries=len(items),
        hit_at_1=hit_at_k(relevance, 1),
        hit_at_5=hit_at_k(relevance, 5),
        hit_at_10=hit_at_k(relevance, 10),
        mrr_at_10=mrr(relevance),
        doc_hit_at_5=hit_at_k(doc_relevance, 5),
        chars_per_token=round(chars / tokens, 3),
        tokenizer=tokenizer_name,
        passages_per_sec=None,
        vram_peak_gb=None,
    )
    return result, records


def _write_outputs(
    run_id: str,
    run_dir: Path,
    results: list[RunResult],
    documents: list[Document],
    items: list[KorQuADItem],
    args: argparse.Namespace,
) -> None:
    gpu = _gpu_state()
    summary = {
        "run_id": run_id,
        "subset": str(args.subset),
        "documents": len(documents),
        "questions": len(items),
        "gpu": gpu,
        "results": [asdict(result) for result in results],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = [
        "model",
        "max_tokens",
        "effective_max_tokens",
        "chunks",
        "hit_at_1",
        "hit_at_5",
        "hit_at_10",
        "mrr_at_10",
        "doc_hit_at_5",
        "chars_per_token",
        "passages_per_sec",
        "vram_peak_gb",
    ]
    rows = [
        {key: value for key, value in asdict(result).items() if value is not None}
        for result in results
    ]
    report = (
        f"# 임베딩 비교 — {run_id}\n\n"
        f"- 서브셋: `{args.subset.name}` 문서 {len(documents):,}개 · KorQuAD dev 질문 {len(items):,}개 "
        "(정답 문서가 서브셋에 있는 질문)\n"
        "- 검색: numpy 전수 비교(정확한 최근접), 정답 판정은 evalkit `contains_answer` (제목 일치 + 공백 무시 정답 포함)\n"
        "- **서브셋 수치는 상대 비교 전용** — 방해 문서가 적어 전체 코퍼스보다 높게 나온다\n"
        f"- 토크나이저: {', '.join(sorted({result.tokenizer for result in results}))}\n"
        f"- GPU: {gpu}\n"
        "- 처리량(`passages_per_sec`)은 이번 실행에서 새로 임베딩한 조합만 표시 (캐시에서 읽으면 빈칸)\n\n"
        + render_table(rows, columns)
    )
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"{run_id}.md").write_text(report, encoding="utf-8")
    print(f"report: {REPORTS_DIR / f'{run_id}.md'}")


def _gpu_state() -> str:
    """처리량 해석용 GPU 상태 (이름 · 전력 상한)."""
    if not torch.cuda.is_available():
        return "cpu"
    query = "--query-gpu=name,enforced.power.limit,clocks.max.sm"
    command = ["nvidia-smi", query, "--format=csv,noheader"]
    # 고정 인자로 PATH 의 nvidia-smi 를 부른다
    smi = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    return smi.stdout.strip() or torch.cuda.get_device_name(0)


if __name__ == "__main__":
    main()
