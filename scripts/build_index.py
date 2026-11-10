#!/usr/bin/env python3
"""Load the catalogue into Postgres with pgvector, embeddings and a lexical index."""
import json, unicodedata, psycopg2
from fastembed import TextEmbedding

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

DDL = """
DROP TABLE IF EXISTS items;
CREATE TABLE items (
  id         int primary key,
  name       text not null,
  descr      text not null,
  search_text text not null,              -- unaccented + lowercased: what people actually type
  cat        text, brand text, variant text,
  embedding  vector(384) not null
);
CREATE INDEX items_tsv  ON items USING gin (to_tsvector('simple', search_text));
CREATE INDEX items_hnsw ON items USING hnsw (embedding vector_cosine_ops)
       WITH (m = 16, ef_construction = 64);
"""

def main():
    rows = json.load(open("out/catalogue.json"))
    emb = TextEmbedding(model_name=MODEL)
    texts = [f"{r['name']}. {r['desc']}" for r in rows]
    vectors = list(emb.embed(texts))            # progress shown by fastembed
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        for r, v in zip(rows, vectors):
            cur.execute(
                "INSERT INTO items (id,name,descr,search_text,cat,brand,variant,embedding)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (r["id"], r["name"], r["desc"],
                 strip_accents(f"{r['name']} {r['desc']}").lower(),
                 r["cat"], r["brand"], r["variant"],
                 "[" + ",".join(f"{x:.6f}" for x in v) + "]"))
        conn.commit()
        cur.execute("SELECT count(*), pg_size_pretty(pg_total_relation_size('items')) FROM items")
        n, size = cur.fetchone()
        cur.execute("SELECT pg_size_pretty(pg_relation_size('items_hnsw'))")
        hnsw = cur.fetchone()[0]
    print(f"loaded {n} rows, table {size}, hnsw index {hnsw}")

if __name__ == "__main__":
    main()
