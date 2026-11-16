"""Benchmark and compare retrieval & context configurations on a given split."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "starter") not in sys.path:
    sys.path.insert(0, str(ROOT / "starter"))

from baseline import hit as any_gold_hit  # noqa: E402
from src.context_builder import pack_context_smart  # noqa: E402
from src.retrievers import (  # noqa: E402
    Bm25Retriever,
    HybridRrfRetriever,
    PgVectorDenseRetriever,
    TfidfRetriever,
)


def strict_groups_hit(chunks_subset: list[dict[str, Any]], question: dict[str, Any]) -> bool:
    """Return True iff EVERY required evidence group in `required_gold_groups` is satisfied."""
    groups = question.get("required_gold_groups") or [question["gold"]]
    return all(any(any_gold_hit(c, group) for c in chunks_subset) for group in groups)


def reciprocal_rank_top3(results_top3: list[dict[str, Any]], gold: list[dict[str, Any]]) -> float:
    for idx, c in enumerate(results_top3, start=1):
        if any_gold_hit(c, gold):
            return 1.0 / idx
    return 0.0


def evaluate_config(
    *,
    config_name: str,
    description: str,
    hypothesis: str,
    retriever: Any,
    corpus_chunks: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    top_k: int,
    budget: int,
    expand_siblings: bool = False,
    header_style: str = "compat",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chunk_by_id = {c["id"]: c for c in corpus_chunks}
    traces: list[dict[str, Any]] = []

    for q in questions:
        t0 = time.perf_counter()
        # Zero gold leakage: ONLY pass q['question'], never q['gold'] or pre-labeled filter
        results = retriever.search(q["question"], filters=None, top_k=top_k)
        context, context_ids = pack_context_smart(
            results,
            budget=budget,
            corpus_chunks=corpus_chunks,
            expand_siblings=expand_siblings,
            header_style=header_style,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        filter_trace = getattr(retriever, "last_filter_trace", None) or {
            "applied_filter": None,
            "filter_reason": None,
            "fallback_used": False,
        }

        marked_results = [
            {
                "id": r["id"],
                "source_id": r["source_id"],
                "source": r["source"],
                "page": r["page"],
                "article": (r.get("metadata") or {}).get("article"),
                "form": (r.get("metadata") or {}).get("form"),
                "page_type": (r.get("metadata") or {}).get("page_type"),
                "rank": r.get("rank", idx + 1),
                "score": round(float(r["score"]), 6),
                "strategy": r.get("strategy", getattr(retriever, "name", config_name)),
                "component_ranks": r.get("component_ranks"),
                "char_count": len(r["text"]),
                "any_hit": any_gold_hit(r, q["gold"]),
            }
            for idx, r in enumerate(results)
        ]

        top1 = results[:1]
        top3 = results[:3]
        ctx_chunks = [chunk_by_id[cid] for cid in context_ids if cid in chunk_by_id]

        any_h1 = bool(top1 and any_gold_hit(top1[0], q["gold"]))
        any_h3 = any(any_gold_hit(c, q["gold"]) for c in top3)
        any_hk = any(any_gold_hit(c, q["gold"]) for c in results)
        any_ctx = any(any_gold_hit(c, q["gold"]) for c in ctx_chunks)

        strict_h1 = strict_groups_hit(top1, q)
        strict_h3 = strict_groups_hit(top3, q)
        strict_hk = strict_groups_hit(results, q)
        strict_ctx = strict_groups_hit(ctx_chunks, q)
        rr3 = reciprocal_rank_top3(top3, q["gold"])

        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "kind": q.get("kind", "direct"),
                "question": q["question"],
                "config": config_name,
                "method": getattr(retriever, "name", config_name),
                "top_k": top_k,
                "budget_chars": budget,
                "expand_siblings": expand_siblings,
                "header_style": header_style,
                "filter_trace": filter_trace,
                "results": marked_results,
                "top3_ids": [r["id"] for r in top3],
                "context_ids": context_ids,
                "context_chars": len(context),
                "est_context_tokens": round(len(context) / 3.5),
                "context": context,
                "elapsed_ms": round(elapsed_ms, 3),
                "strict_hit1": strict_h1,
                "strict_hit3": strict_h3,
                "strict_hit_topk": strict_hk,
                "strict_context_hit": strict_ctx,
                "any_hit1": any_h1,
                "any_hit3": any_h3,
                "any_hit_topk": any_hk,
                "any_context_hit": any_ctx,
                "rr3": round(rr3, 4),
            }
        )

    n = len(traces)
    summary = {
        "config": config_name,
        "method": getattr(retriever, "name", config_name),
        "description": description,
        "hypothesis": hypothesis,
        "split": questions[0]["split"] if questions else "",
        "n": n,
        "top_k": top_k,
        "budget_chars": budget,
        "expand_siblings": expand_siblings,
        "header_style": header_style,
        "strict_hit1": sum(t["strict_hit1"] for t in traces),
        "strict_hit1_rate": round(sum(t["strict_hit1"] for t in traces) / n, 4),
        "strict_hit3": sum(t["strict_hit3"] for t in traces),
        "strict_hit3_rate": round(sum(t["strict_hit3"] for t in traces) / n, 4),
        "strict_hit_topk": sum(t["strict_hit_topk"] for t in traces),
        "strict_context_hit": sum(t["strict_context_hit"] for t in traces),
        "strict_context_rate": round(sum(t["strict_context_hit"] for t in traces) / n, 4),
        "any_hit1": sum(t["any_hit1"] for t in traces),
        "any_hit3": sum(t["any_hit3"] for t in traces),
        "any_context_hit": sum(t["any_context_hit"] for t in traces),
        "mrr3": round(statistics.mean(t["rr3"] for t in traces), 4),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
        "avg_est_tokens": round(statistics.mean(t["est_context_tokens"] for t in traces), 1),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2),
        "mean_ms": round(statistics.mean(t["elapsed_ms"] for t in traces), 2),
        "queries_with_inferred_filter": [
            {"id": t["query_id"], "filter": t["filter_trace"]["applied_filter"]}
            for t in traces
            if t["filter_trace"]["applied_filter"]
        ],
        "strict_missed_hit3_ids": [t["query_id"] for t in traces if not t["strict_hit3"]],
        "strict_missed_context_ids": [t["query_id"] for t in traces if not t["strict_context_hit"]],
        "load_seconds": round(float(getattr(retriever, "load_seconds", 0.0)), 3),
        "service_cost_usd": 0.0,
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Day 7 retrieval & context configs.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Safety flag: only pass after dev configuration selection is frozen.",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional subset of config names to run (e.g., for locked test comparison).",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "retrieval")
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        raise RuntimeError(
            "Chặn chạy tập test khi chưa bật --allow-test. Hãy hoàn tất thử nghiệm và chốt cấu hình trên dev trước!"
        )

    load_dotenv(ROOT / ".env")
    db_url = os.environ["DATABASE_URL"]
    db_strategy = os.environ.get("DAY6_STRATEGY", "B_para")
    emb_model = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

    corpus_chunks = json.loads((args.data_dir / "chunks.json").read_text(encoding="utf-8"))
    all_questions = json.loads((args.data_dir / "questions.json").read_text(encoding="utf-8"))
    selected_questions = [q for q in all_questions if q["split"] == args.split]
    if not selected_questions:
        raise ValueError(f"Không có câu hỏi nào cho split={args.split}")

    tfidf = TfidfRetriever(corpus_chunks)
    bm25 = Bm25Retriever(corpus_chunks, use_bigrams=True)
    dense = PgVectorDenseRetriever(
        db_url,
        chunks_for_metadata=corpus_chunks,
        db_strategy=db_strategy,
        model_name=emb_model,
    )
    hybrid_plain = HybridRrfRetriever(dense, bm25, candidate_k=10, rrf_k=60, auto_infer_filter=False)
    hybrid_qfilter = HybridRrfRetriever(dense, bm25, candidate_k=10, rrf_k=60, auto_infer_filter=True)
    hybrid_weighted_qfilter = HybridRrfRetriever(
        dense,
        bm25,
        candidate_k=10,
        rrf_k=60,
        dense_weight=2.0,
        bm25_weight=1.0,
        auto_infer_filter=True,
    )

    specs = [
        {
            "config_name": "cfg1_tfidf_b800",
            "description": "Baseline Lexical TF-IDF (unigram), top_k=3, no filter, budget=800 chars",
            "hypothesis": "Baseline mặc định từ starter để làm mốc so sánh.",
            "retriever": tfidf,
            "top_k": 3,
            "budget": 800,
            "expand_siblings": False,
            "header_style": "compat",
        },
        {
            "config_name": "cfg2_bm25_b800",
            "description": "Phương pháp 2: Okapi BM25 (unigram + Vietnamese word-bigram), top_k=3, no filter, budget=800 chars",
            "hypothesis": "Chuẩn hóa độ dài đoạn và thêm cụm từ ghép 2 âm tiết tiếng Việt sẽ cải thiện xếp hạng lexical so với TF-IDF thuần.",
            "retriever": bm25,
            "top_k": 3,
            "budget": 800,
            "expand_siblings": False,
            "header_style": "compat",
        },
        {
            "config_name": "cfg3_dense_b800",
            "description": "Phương pháp 3: Dense Retrieval (multilingual-e5-base + pgvector Day 6), top_k=3, no filter, budget=800 chars",
            "hypothesis": "Biểu diễn ngữ nghĩa E5-base sẽ xử lý tốt các câu hỏi diễn đạt lại (D02, D05, D14) mà lexical bị trượt.",
            "retriever": dense,
            "top_k": 3,
            "budget": 800,
            "expand_siblings": False,
            "header_style": "compat",
        },
        {
            "config_name": "cfg4_hybrid_rrf_b800",
            "description": "Phương pháp 4: Hybrid RRF (Dense pgvector + BM25, k_rrf=60, cand=10), top_k=3, no filter, budget=800 chars",
            "hypothesis": "Kết hợp ngữ nghĩa của E5 và khớp từ khóa/số hiệu chính xác của BM25 bằng RRF sẽ ổn định cả Hit@1 lẫn Hit@3.",
            "retriever": hybrid_plain,
            "top_k": 3,
            "budget": 800,
            "expand_siblings": False,
            "header_style": "compat",
        },
        {
            "config_name": "imp1_hybrid_qfilter_b2400",
            "description": "Cải thiện 1: Hybrid RRF + Suy luận Filter từ Query (article/page_type) + Context Budget=2400 chars (rich header)",
            "hypothesis": (
                "Khắc phục 2 lỗi dev: (1) Nâng budget từ 800 lên 2400 chars để không cắt rỗng chunk dài >720 ký tự ở D06, D07 "
                "và giữ đủ 2 đoạn cho câu multi-hop D17; (2) Tự suy luận filter từ query ('tại Phụ lục', 'Theo Điều N') để loại nhiễu thân văn bản ở D15, D16."
            ),
            "retriever": hybrid_qfilter,
            "top_k": 3,
            "budget": 2400,
            "expand_siblings": False,
            "header_style": "rich",
        },
        {
            "config_name": "imp2_hybrid_sibling_b2800",
            "description": "Cải thiện 2: Hybrid RRF (1:1) + Suy luận Filter + top_k=4 + Mở rộng chunk kế cận cùng Điều/cùng trang Phụ lục, Budget=2800 chars",
            "hypothesis": (
                "Bổ sung các chunk kế cận cùng Điều hoặc phần cuối/chú thích cùng trang Phụ lục sau các hit chính giúp bao phủ trọn điều kiện/ngoại lệ "
                "hoặc chú thích biểu mẫu bị tách chunk, đổi lại context tăng thêm ~400-700 ký tự."
            ),
            "retriever": hybrid_qfilter,
            "top_k": 4,
            "budget": 2800,
            "expand_siblings": True,
            "header_style": "rich",
        },
        {
            "config_name": "imp3_whyb_qfilter_sibling_b2800",
            "description": "Cải thiện 3: Weighted Hybrid RRF (Dense=2.0, BM25=1.0) + Suy luận Filter + top_k=4 + Sibling Expansion, Budget=2800 chars",
            "hypothesis": (
                "Tăng trọng số Dense gấp đôi BM25 (2:1) trong RRF để khắc phục lỗi BM25 kéo tụt hạng 1 của câu diễn đạt lại (D02, D05), "
                "đồng thời giữ nguyên khả năng bao phủ context 100% của Sibling Expansion."
            ),
            "retriever": hybrid_weighted_qfilter,
            "top_k": 4,
            "budget": 2800,
            "expand_siblings": True,
            "header_style": "rich",
        },
    ]

    if args.configs:
        wanted = set(args.configs)
        specs = [s for s in specs if s["config_name"] in wanted]
        if not specs:
            raise ValueError(f"Không tìm thấy cấu hình khớp: {args.configs}")

    summaries: list[dict[str, Any]] = []
    split_dir = args.output_dir / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        summary, traces = evaluate_config(
            corpus_chunks=corpus_chunks,
            questions=selected_questions,
            **spec,
        )
        cfg_dir = split_dir / spec["config_name"]
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (cfg_dir / "traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)

    comp_path = args.output_dir / f"{args.split}_comparison.json"
    comp_path.write_text(
        json.dumps({"split": args.split, "summaries": summaries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"=== Retrieval & Context Comparison ({args.split}, N={len(selected_questions)}) ===")
    for s in summaries:
        print(
            f"{s['config']:27s} | Strict H@1={s['strict_hit1']:2d}/{s['n']} | "
            f"Strict H@3={s['strict_hit3']:2d}/{s['n']} | "
            f"Strict Ctx={s['strict_context_hit']:2d}/{s['n']} "
            f"(Any Ctx={s['any_context_hit']:2d}/{s['n']}) | "
            f"MRR@3={s['mrr3']:.3f} | chars={s['avg_context_chars']:6.1f} | "
            f"med_ms={s['median_ms']:6.2f} | missed_ctx={s['strict_missed_context_ids']}"
        )


if __name__ == "__main__":
    main()
