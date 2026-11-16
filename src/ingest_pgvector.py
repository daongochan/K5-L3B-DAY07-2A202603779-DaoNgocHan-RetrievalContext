"""Self-contained helper to recreate the Day 6 pgvector `chunks` table from `data/chunks.json`.

If you already have the Day 6 container (`day6_student_starter-postgres-1`) running on port 5433,
you do not need to run this script. Graders cloning only this repository can run:
    docker compose up -d
    python -m src.ingest_pgvector
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DOCUMENT_PREFIX = "passage: "


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate pgvector from data/chunks.json using multilingual-e5-base.")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data" / "chunks.json")
    parser.add_argument("--strategy", default=None)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    db_url = os.environ["DATABASE_URL"]
    strategy = args.strategy or os.environ.get("DAY6_STRATEGY", "B_para")
    model_name = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    dim_fn = getattr(model, "get_embedding_dimension", None) or getattr(model, "get_sentence_embedding_dimension")
    dims = int(dim_fn())
    vectors = model.encode(
        [DOCUMENT_PREFIX + c["text"] for c in chunks],
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=False,
    ).tolist()
    embed_sec = time.perf_counter() - t0

    with psycopg.connect(db_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id     text PRIMARY KEY,
                    strategy     text NOT NULL,
                    document_id  text NOT NULL,
                    source       text NOT NULL,
                    page         int  NOT NULL,
                    printed_page int,
                    page_type    text,
                    article      text,
                    chapter      text,
                    section      text,
                    form         text,
                    text         text NOT NULL,
                    char_count   int  NOT NULL,
                    embedding    vector({dims}) NOT NULL
                )
                """
            )
            ids = [c["id"] for c in chunks]
            cur.execute("DELETE FROM chunks WHERE strategy = %s AND chunk_id <> ALL(%s)", (strategy, ids))
            rows = [
                (
                    c["id"],
                    strategy,
                    c["source_id"],
                    c["source"],
                    int(c["page"]),
                    (c.get("metadata") or {}).get("printed_page"),
                    (c.get("metadata") or {}).get("page_type"),
                    (c.get("metadata") or {}).get("article"),
                    (c.get("metadata") or {}).get("chapter"),
                    (c.get("metadata") or {}).get("section"),
                    (c.get("metadata") or {}).get("form"),
                    c["text"],
                    int((c.get("metadata") or {}).get("char_count", len(c["text"]))),
                    vec,
                )
                for c, vec in zip(chunks, vectors)
            ]
            cur.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, strategy, document_id, source, page, printed_page,
                    page_type, article, chapter, section, form, text, char_count, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    strategy     = EXCLUDED.strategy,
                    document_id  = EXCLUDED.document_id,
                    source       = EXCLUDED.source,
                    page         = EXCLUDED.page,
                    printed_page = EXCLUDED.printed_page,
                    page_type    = EXCLUDED.page_type,
                    article      = EXCLUDED.article,
                    chapter      = EXCLUDED.chapter,
                    section      = EXCLUDED.section,
                    form         = EXCLUDED.form,
                    text         = EXCLUDED.text,
                    char_count   = EXCLUDED.char_count,
                    embedding    = EXCLUDED.embedding
                """,
                rows,
            )
        conn.commit()

    print(f"Ingested {len(rows)} chunks (strategy={strategy}, dims={dims}) in {embed_sec:.1f}s.")


if __name__ == "__main__":
    main()
