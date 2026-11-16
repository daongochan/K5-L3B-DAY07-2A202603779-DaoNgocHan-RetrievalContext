"""Context selection, deduplication, sibling expansion, and budget packing for Day 7."""
from __future__ import annotations

from typing import Any


def format_chunk_block(chunk: dict[str, Any], *, style: str = "compat") -> str:
    """Format a single chunk with source provenance for LLM grounding & verification.

    - `compat`: Matches `starter/baseline.py` (`[id | source | page X]\\ntext`) so character budgets
      are 100% comparable with baseline when `style='compat'`.
    - `rich`: Includes printed Công báo page and Article/Form metadata in header (`[id | source | page X (in Y) | Điều Z]\\ntext`).
    """
    cid = chunk["id"]
    src = chunk["source"]
    page = chunk["page"]
    if style == "compat":
        return f"[{cid} | {src} | page {page}]\n{chunk['text']}"

    meta = chunk.get("metadata") or {}
    printed = meta.get("printed_page")
    page_label = f"page {page}" + (f" (Công báo tr.{printed})" if printed else "")
    scope_parts = [p for p in (meta.get("article"), meta.get("form")) if p]
    scope_str = f" | {' - '.join(scope_parts)}" if scope_parts else ""
    return f"[{cid} | {src} | {page_label}{scope_str}]\n{chunk['text']}"


def expand_adjacent_siblings(
    primary_results: list[dict[str, Any]],
    corpus_chunks: list[dict[str, Any]],
    *,
    anchor_limit: int = 2,
    max_siblings_per_anchor: int = 2,
) -> list[dict[str, Any]]:
    """Append adjacent same-article or same-appendix-page chunks AFTER primary retrieved chunks.

    Keeping primary `top_k` chunks first guarantees multi-hop second/third hits are never
    displaced by neighbor chunks of rank-1, while still recovering split clauses or form footnotes.
    """
    if not primary_results or not corpus_chunks:
        return list(primary_results)

    idx_by_id = {c["id"]: i for i, c in enumerate(corpus_chunks)}
    seen = {c["id"] for c in primary_results}
    extras: list[dict[str, Any]] = []

    for anchor in primary_results[:anchor_limit]:
        pos = idx_by_id.get(anchor["id"])
        if pos is None:
            continue
        a_meta = anchor.get("metadata") or {}
        a_article = a_meta.get("article")
        a_page_type = a_meta.get("page_type")
        a_page = anchor.get("page")

        added_for_anchor = 0
        # For appendix forms, check same-page chunks (especially the closing/footnote chunk at end of page)
        if a_page_type == "phu_luc_bieu_mau":
            same_page = [
                c for c in corpus_chunks if c.get("page") == a_page and c["id"] not in seen
            ]
            # Prioritize closing chunk (contains commitment + footnotes) then immediate next chunk
            ordered_candidates = list(reversed(same_page))
            for cand in ordered_candidates:
                if cand["id"] not in seen:
                    seen.add(cand["id"])
                    extras.append({**cand, "score": 0.0, "strategy": "sibling_expansion"})
                    added_for_anchor += 1
                    if added_for_anchor >= max_siblings_per_anchor:
                        break
        elif a_article:
            # Immediate next then immediate previous chunk if they belong to the exact same Article
            for neighbor_idx in (pos + 1, pos - 1, pos + 2):
                if 0 <= neighbor_idx < len(corpus_chunks):
                    cand = corpus_chunks[neighbor_idx]
                    c_meta = cand.get("metadata") or {}
                    if c_meta.get("article") == a_article and cand["id"] not in seen:
                        seen.add(cand["id"])
                        extras.append({**cand, "score": 0.0, "strategy": "sibling_expansion"})
                        added_for_anchor += 1
                        if added_for_anchor >= max_siblings_per_anchor:
                            break

    return list(primary_results) + extras


def interleave_search_batches(batches: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Round-robin interleave across multiple search turns (rank 1 of each turn, then rank 2...)
    so later agent searches are never starved by rank-2/3 chunks from turn 1.
    """
    if not batches:
        return []
    max_len = max((len(b) for b in batches), default=0)
    interleaved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r_idx in range(max_len):
        # Prioritize newer targeted searches slightly or preserve chronological rank-1s:
        # chronological rank-1 of turn 1, rank-1 of turn 2, ..., then rank-2 of turn 1, rank-2 of turn 2...
        for batch in batches:
            if r_idx < len(batch):
                chunk = batch[r_idx]
                if chunk["id"] not in seen:
                    seen.add(chunk["id"])
                    interleaved.append(chunk)
    return interleaved


def pack_context_smart(
    candidates: list[dict[str, Any]],
    *,
    budget: int = 2400,
    corpus_chunks: list[dict[str, Any]] | None = None,
    expand_siblings: bool = False,
    header_style: str = "compat",
) -> tuple[str, list[str]]:
    """Deduplicate, optionally expand siblings, and greedily pack chunks within `budget` chars."""
    if budget < 1:
        raise ValueError("budget phải >= 1")

    ordered = (
        expand_adjacent_siblings(candidates, corpus_chunks)
        if (expand_siblings and corpus_chunks is not None)
        else list(candidates)
    )

    blocks: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    current_len = 0

    for chunk in ordered:
        cid = chunk["id"]
        if cid in seen:
            continue
        seen.add(cid)
        block = format_chunk_block(chunk, style=header_style)
        sep_len = 2 if blocks else 0
        if current_len + sep_len + len(block) <= budget:
            blocks.append(block)
            ids.append(cid)
            current_len += sep_len + len(block)

    return "\n\n".join(blocks), ids
