"""Retrieval methods for Day 7: TF-IDF, BM25 (Vietnamese unigram+bigram), Dense pgvector (E5), and Hybrid RRF."""
from __future__ import annotations

import math
import os
import re
import time
import unicodedata
from collections import Counter
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
QUERY_PREFIX = "query: "

ALLOWED_FILTER_FIELDS = {
    "source_id": "document_id",
    "document_id": "document_id",
    "page": "page",
    "printed_page": "printed_page",
    "page_type": "page_type",
    "article": "article",
    "chapter": "chapter",
    "section": "section",
    "form": "form",
}

_EMBEDDER_CACHE: dict[str, tuple[Any, float]] = {}


def tokenize_vi(text: str, *, use_bigrams: bool = False) -> list[str]:
    """Normalize NFC, lowercase, extract word tokens and optional adjacent word bigrams."""
    norm = unicodedata.normalize("NFC", text).casefold()
    words = re.findall(r"\w+", norm)
    if not use_bigrams or len(words) < 2:
        return words
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


def infer_query_filters(query: str) -> tuple[dict[str, Any] | None, str | None]:
    """Infer high-precision metadata filters solely from the user's query text (zero gold leakage).

    Rules:
    1. Appendix filter (`page_type = 'phu_luc_bieu_mau'`):
       Triggered when the query explicitly asks about the Appendix (`tại Phụ lục`, `trong Phụ lục`)
       and is not negated/excluded (`ngoài Phụ lục`, `trừ Phụ lục`).
    2. Single-Article filter (`article = 'Điều N'`):
       Triggered ONLY when:
       - Exactly ONE distinct article number (1..44) appears in the entire query, AND
       - It is anchored by a scope phrase (`theo Điều N`, `tại Điều N`, `Điều N quy định`, `quy định tại Điều N`)
         NOT preceded by exclusion/comparison words (`ngoài`, `trừ`, `khác`, `so với`, `so sánh`).
    """
    norm = unicodedata.normalize("NFC", query)
    low = norm.casefold()

    # Check Appendix scope
    if re.search(r"\b(tại|trong|ở)\s+phụ\s+lục\b", low) and not re.search(
        r"\b(ngoài|trừ)\s+(các\s+)?(quy\s+định\s+)?(tại\s+)?phụ\s+lục\b", low
    ):
        return (
            {"page_type": "phu_luc_bieu_mau"},
            "Query nêu rõ phạm vi 'tại/trong Phụ lục' -> lọc page_type='phu_luc_bieu_mau'.",
        )

    # Check explicit single-Article scope
    all_articles = {int(m.group(1)) for m in re.finditer(r"\bđiều\s+(\d{1,2})\b", low)}
    valid_articles = {n for n in all_articles if 1 <= n <= 44}
    if len(valid_articles) == 1:
        art_num = next(iter(valid_articles))
        has_exclusion = bool(
            re.search(
                rf"\b(ngoài|trừ|khác|so\s+với|so\s+sánh)[^.;,?]*\bđiều\s+{art_num}\b",
                low,
            )
        )
        has_scope_anchor = bool(
            re.search(
                rf"(\b(theo|tại)\s+điều\s+{art_num}\b|\bđiều\s+{art_num}\s+(nghị\s+định\s+[^,;]*\s+)?quy\s+định\b)",
                low,
            )
        )
        if has_scope_anchor and not has_exclusion:
            return (
                {"article": f"Điều {art_num}"},
                f"Query giới hạn tường minh vào một điều duy nhất ('Điều {art_num}') -> lọc article='Điều {art_num}'.",
            )

    return None, None


def chunk_matches_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Check metadata filter match on an in-memory Day 7 chunk dict."""
    if not filters:
        return True
    meta = chunk.get("metadata") or {}
    for key, expected in filters.items():
        if key not in ALLOWED_FILTER_FIELDS:
            raise ValueError(f"Trường filter không được hỗ trợ: {key}")
        if key in ("source_id", "document_id"):
            actual = chunk.get("source_id")
        elif key == "page":
            actual = chunk.get("page")
        else:
            actual = meta.get(key)
        if actual != expected:
            return False
    return True


class TfidfRetriever:
    """Method 1 (Baseline): Unigram TF-IDF cosine similarity."""

    name = "tfidf"

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        t0 = time.perf_counter()
        self.chunks = chunks
        counts = [Counter(tokenize_vi(c["text"], use_bigrams=False)) for c in chunks]
        df = Counter(term for row in counts for term in row)
        n_docs = len(chunks)
        self.idf = {
            term: math.log((1 + n_docs) / (1 + freq)) + 1.0
            for term, freq in df.items()
        }
        self.vectors = [self._to_unit_vec(row) for row in counts]
        self.load_seconds = time.perf_counter() - t0

    def _to_unit_vec(self, counts: Counter[str]) -> dict[str, float]:
        vec = {t: c * self.idf[t] for t, c in counts.items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not query.strip() or top_k < 1:
            return []
        q_vec = self._to_unit_vec(Counter(tokenize_vi(query, use_bigrams=False)))
        scored: list[dict[str, Any]] = []
        for chunk, d_vec in zip(self.chunks, self.vectors):
            if not chunk_matches_filters(chunk, filters):
                continue
            score = sum(qv * d_vec.get(term, 0.0) for term, qv in q_vec.items())
            if score > 0:
                scored.append({**chunk, "score": float(score), "strategy": self.name})
        scored.sort(key=lambda c: (-c["score"], c["id"]))
        return [
            {**item, "rank": idx + 1}
            for idx, item in enumerate(scored[:top_k])
        ]


class Bm25Retriever:
    """Method 2: Okapi BM25 with Vietnamese word unigrams + adjacent word bigrams."""

    name = "bm25_vn"

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        use_bigrams: bool = True,
    ) -> None:
        t0 = time.perf_counter()
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.use_bigrams = use_bigrams

        self.doc_tokens = [tokenize_vi(c["text"], use_bigrams=use_bigrams) for c in chunks]
        self.doc_freqs = [Counter(toks) for toks in self.doc_tokens]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0

        n_docs = len(chunks)
        df = Counter(term for freqs in self.doc_freqs for term in freqs)
        # Robertson-Sparck Jones smoothed non-negative IDF
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self.load_seconds = time.perf_counter() - t0

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not query.strip() or top_k < 1:
            return []
        q_terms = Counter(tokenize_vi(query, use_bigrams=self.use_bigrams))
        scored: list[dict[str, Any]] = []
        for chunk, freqs, dl in zip(self.chunks, self.doc_freqs, self.doc_lens):
            if not chunk_matches_filters(chunk, filters):
                continue
            score = 0.0
            norm_denom_base = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
            for term, qf in q_terms.items():
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                tf_norm = (tf * (self.k1 + 1.0)) / (tf + norm_denom_base)
                # Slight weight boost for exact 2-word Vietnamese compound phrase hits
                phrase_weight = 1.25 if "_" in term else 1.0
                score += idf * tf_norm * phrase_weight * (1.0 + 0.1 * (qf - 1))
            if score > 0:
                scored.append({**chunk, "score": float(score), "strategy": self.name})
        scored.sort(key=lambda c: (-c["score"], c["id"]))
        return [
            {**item, "rank": idx + 1}
            for idx, item in enumerate(scored[:top_k])
        ]


def _get_cached_embedder(model_name: str) -> tuple[Any, float]:
    if model_name not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer

        t0 = time.perf_counter()
        try:
            model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            model = SentenceTransformer(model_name)
        elapsed = time.perf_counter() - t0
        _EMBEDDER_CACHE[model_name] = (model, elapsed)
    return _EMBEDDER_CACHE[model_name]


class PgVectorDenseRetriever:
    """Method 3: Real Dense Retrieval on Day 6 PostgreSQL + pgvector with multilingual-e5-base."""

    name = "dense_pgvector"

    def __init__(
        self,
        database_url: str,
        *,
        chunks_for_metadata: list[dict[str, Any]] | None = None,
        db_strategy: str = "B_para",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        if not database_url:
            raise ValueError("database_url không được rỗng")
        self.database_url = database_url
        self.db_strategy = db_strategy
        self.model_name = model_name
        self._meta_by_id = {c["id"]: c.get("metadata", {}) for c in (chunks_for_metadata or [])}

        model, load_sec = _get_cached_embedder(model_name)
        self._model = model
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self.dimensions = int(get_dim())
        self.load_seconds = load_sec

        # Verify live connection up front so failures are never hidden
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE strategy = %s", (self.db_strategy,))
            count = int(cur.fetchone()[0])
            if count == 0:
                raise RuntimeError(f"Không tìm thấy chunk nào trong pgvector cho strategy={self.db_strategy!r}")
            self.indexed_chunks = count

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.database_url)
        register_vector(conn)
        return conn

    def embed_query(self, query: str) -> list[float]:
        vec = self._model.encode(
            QUERY_PREFIX + query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.tolist()

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not query.strip() or top_k < 1:
            return []
        q_vec = self.embed_query(query)

        clauses = ["strategy = %s"]
        params: list[Any] = [self.db_strategy]
        for key, value in (filters or {}).items():
            if key not in ALLOWED_FILTER_FIELDS:
                raise ValueError(f"Trường filter không được hỗ trợ: {key}")
            col = ALLOWED_FILTER_FIELDS[key]
            if value is None:
                clauses.append(f'"{col}" IS NULL')
            else:
                clauses.append(f'"{col}" = %s')
                params.append(value)

        where_sql = " AND ".join(clauses)
        sql = f"""
            SELECT chunk_id, document_id, source, page, printed_page, page_type,
                   article, chapter, section, form, text, char_count,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector, chunk_id ASC
            LIMIT %s
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, [q_vec, *params, q_vec, top_k])
            rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            cid = row[0]
            base_meta = dict(self._meta_by_id.get(cid, {}))
            base_meta.update(
                {
                    "printed_page": row[4],
                    "page_type": row[5],
                    "article": row[6],
                    "chapter": row[7],
                    "section": row[8],
                    "form": row[9],
                    "char_count": row[11],
                }
            )
            results.append(
                {
                    "id": cid,
                    "source_id": row[1],
                    "source": row[2],
                    "page": int(row[3]),
                    "text": row[10],
                    "metadata": base_meta,
                    "score": float(row[12]),
                    "rank": idx + 1,
                    "strategy": self.name,
                }
            )
        return results


def fuse_rrf(
    ranked_lists: list[tuple[str, list[dict[str, Any]], float]],
    *,
    rrf_k: int = 60,
    top_k: int = 3,
    strategy_name: str = "hybrid_rrf",
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion over multiple ranked lists: sum(weight / (rrf_k + rank))."""
    by_id: dict[str, dict[str, Any]] = {}
    rrf_scores: dict[str, float] = {}
    component_ranks: dict[str, dict[str, Any]] = {}

    for source_name, items, weight in ranked_lists:
        for rank_idx, item in enumerate(items, start=1):
            cid = item["id"]
            if cid not in by_id:
                by_id[cid] = item
                rrf_scores[cid] = 0.0
                component_ranks[cid] = {}
            rrf_scores[cid] += weight / (rrf_k + rank_idx)
            component_ranks[cid][source_name] = {
                "rank": rank_idx,
                "raw_score": round(float(item["score"]), 6),
            }

    merged: list[dict[str, Any]] = []
    for cid, base in by_id.items():
        meta = dict(base.get("metadata") or {})
        merged.append(
            {
                **base,
                "metadata": meta,
                "score": float(rrf_scores[cid]),
                "component_ranks": component_ranks[cid],
                "strategy": strategy_name,
            }
        )
    merged.sort(key=lambda c: (-c["score"], c["id"]))
    return [
        {**item, "rank": idx + 1}
        for idx, item in enumerate(merged[:top_k])
    ]


class HybridRrfRetriever:
    """Method 4: Hybrid RRF combining Dense (pgvector E5) + Lexical (BM25 Vietnamese).

    Supports optional `auto_infer_filter=True` which derives reliable metadata filters
    solely from user query text, with safe fallback fill if filtered candidates < top_k.
    """

    name = "hybrid_rrf"

    def __init__(
        self,
        dense: PgVectorDenseRetriever,
        bm25: Bm25Retriever,
        *,
        candidate_k: int = 10,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        auto_infer_filter: bool = False,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.auto_infer_filter = auto_infer_filter
        self.load_seconds = dense.load_seconds + bm25.load_seconds
        self.last_filter_trace: dict[str, Any] = {}

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not query.strip() or top_k < 1:
            return []

        effective_filters = dict(filters) if filters else None
        filter_reason = "user_explicit_filter" if effective_filters else None

        if effective_filters is None and self.auto_infer_filter:
            inferred, reason = infer_query_filters(query)
            if inferred:
                effective_filters = inferred
                filter_reason = reason

        cand_k = max(self.candidate_k, top_k * 2)
        dense_hits = self.dense.search(query, filters=effective_filters, top_k=cand_k)
        bm25_hits = self.bm25.search(query, filters=effective_filters, top_k=cand_k)
        fused = fuse_rrf(
            [
                ("dense_pgvector", dense_hits, self.dense_weight),
                ("bm25_vn", bm25_hits, self.bm25_weight),
            ],
            rrf_k=self.rrf_k,
            top_k=top_k,
            strategy_name=self.name,
        )

        fallback_used = False
        if effective_filters and len(fused) < top_k:
            # Safe fallback: top up remaining slots from unfiltered hybrid search without duplicates
            fallback_used = True
            dense_unfiltered = self.dense.search(query, filters=None, top_k=cand_k)
            bm25_unfiltered = self.bm25.search(query, filters=None, top_k=cand_k)
            unfiltered_fused = fuse_rrf(
                [
                    ("dense_pgvector", dense_unfiltered, self.dense_weight),
                    ("bm25_vn", bm25_unfiltered, self.bm25_weight),
                ],
                rrf_k=self.rrf_k,
                top_k=cand_k,
                strategy_name=self.name,
            )
            seen = {c["id"] for c in fused}
            for item in unfiltered_fused:
                if item["id"] not in seen:
                    fused.append(item)
                    seen.add(item["id"])
                    if len(fused) >= top_k:
                        break
            fused = [{**it, "rank": i + 1} for i, it in enumerate(fused[:top_k])]

        self.last_filter_trace = {
            "applied_filter": effective_filters,
            "filter_reason": filter_reason,
            "fallback_used": fallback_used,
        }
        return fused
