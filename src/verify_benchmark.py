"""Verify integrity, split isolation, evidence coverage, and pgvector sync for Day 7 benchmark."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "starter") not in sys.path:
    sys.path.insert(0, str(ROOT / "starter"))

from baseline import hit, validate as baseline_validate  # noqa: E402


def verify_files(data_dir: Path, check_db: bool = True) -> dict[str, Any]:
    chunks = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
    questions = json.loads((data_dir / "questions.json").read_text(encoding="utf-8"))
    agent_cases = json.loads((data_dir / "agent_cases.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))

    # 1. Starter contract validation on all answerable questions
    baseline_validate(chunks, questions)

    # 2. Verify required_gold_groups and zero pre-baked gold filter leakage
    for q in questions:
        if "filter" in q and q["filter"] is not None:
            raise ValueError(f"{q['id']}: không gắn sẵn filter từ nhãn chuẩn vào questions.json")
        groups = q.get("required_gold_groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError(f"{q['id']}: thiếu required_gold_groups")
        for idx, group in enumerate(groups):
            if not any(hit(c, group) for c in chunks):
                raise ValueError(f"{q['id']}: nhóm evidence bắt buộc #{idx} không khớp chunk nào")

    # 3. Verify split counts against BENCHMARK.md thresholds
    dev_ans = [q for q in questions if q["split"] == "dev"]
    test_ans = [q for q in questions if q["split"] == "test"]
    dev_unans = [c for c in agent_cases if c["split"] == "dev" and not c["answerable"]]
    test_unans = [c for c in agent_cases if c["split"] == "test" and not c["answerable"]]

    if len(dev_ans) < 15 or len(dev_unans) < 2:
        raise ValueError("Dev chưa đủ tối thiểu 15 câu có đáp án và 2 câu thiếu bằng chứng")
    if len(test_ans) < 5 or len(test_unans) < 2:
        raise ValueError("Test chưa đủ tối thiểu 5 câu có đáp án và 2 câu thiếu bằng chứng")
    if len(agent_cases) != len(questions) + len(dev_unans) + len(test_unans):
        raise ValueError("Tổng số agent_cases không khớp questions + unanswerable")

    # 4. Verify unanswerable case fields & no gold leakage in agent_cases
    seen_ids: set[str] = set()
    seen_qtexts: set[str] = set()
    for c in agent_cases:
        cid = c["id"]
        if cid in seen_ids:
            raise ValueError(f"Trùng ID trong agent_cases: {cid}")
        seen_ids.add(cid)
        norm_q = " ".join(c["question"].lower().split())
        if norm_q in seen_qtexts:
            raise ValueError(f"Trùng nội dung câu hỏi trong agent_cases: {cid}")
        seen_qtexts.add(norm_q)
        if "gold" in c or "expected_answer" in c:
            raise ValueError(f"{cid}: agent_cases không được chứa gold hoặc expected_answer")
        if not c["answerable"]:
            if c.get("expected_behavior") != "search_and_abstain":
                raise ValueError(f"{cid}: expected_behavior phải là search_and_abstain")
            if not isinstance(c.get("absence_reason"), str) or len(c["absence_reason"].strip()) < 20:
                raise ValueError(f"{cid}: thiếu absence_reason giải thích cách kiểm tra tài liệu")

    # 5. Check page overlap between dev and test answerable gold
    dev_pages = {g["page"] for q in dev_ans for g in q["gold"]}
    test_pages = {g["page"] for q in test_ans for g in q["gold"]}
    shared_pages = sorted(dev_pages & test_pages)

    # 6. Optional live pgvector check
    db_status: dict[str, Any] = {"checked": False}
    if check_db:
        from dotenv import load_dotenv
        import psycopg

        load_dotenv(ROOT / ".env")
        db_url = os.environ.get("DATABASE_URL")
        strategy = os.environ.get("DAY6_STRATEGY", "B_para")
        if not db_url:
            raise RuntimeError("Thiếu DATABASE_URL để kiểm tra pgvector")
        with psycopg.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, document_id, page, text FROM chunks WHERE strategy = %s",
                (strategy,),
            )
            rows = {r[0]: r for r in cur.fetchall()}
        if len(rows) != len(chunks):
            raise ValueError(f"Số chunk trên DB ({len(rows)}) khác chunks.json ({len(chunks)})")
        for c in chunks:
            row = rows.get(c["id"])
            if row is None or row[1] != c["source_id"] or row[2] != c["page"] or row[3] != c["text"]:
                raise ValueError(f"Chunk {c['id']} trên DB không khớp chunks.json")
        db_status = {"checked": True, "strategy": strategy, "matched_chunks": len(rows)}

    return {
        "status": "OK",
        "chunks": len(chunks),
        "content_pages_in_manifest": manifest["inherited_from_day6"]["corpus_document"]["content_pages"],
        "dev": {"answerable": len(dev_ans), "unanswerable": len(dev_unans), "total": len(dev_ans) + len(dev_unans)},
        "test": {"answerable": len(test_ans), "unanswerable": len(test_unans), "total": len(test_ans) + len(test_unans)},
        "total_cases": len(agent_cases),
        "multi_evidence_questions": [q["id"] for q in questions if len(q["required_gold_groups"]) > 1],
        "shared_gold_pages_between_dev_and_test": shared_pages,
        "pgvector_sync": db_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Day 7 data and benchmark integrity.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--no-db", action="store_true", help="Skip live PostgreSQL check")
    args = parser.parse_args()

    summary = verify_files(args.data_dir, check_db=not args.no_db)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
