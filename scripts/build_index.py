#!/usr/bin/env python3
"""Load the catalogue into Postgres with pgvector, embeddings and a lexical index.

usage: build_index.py <table> <model>
Embedding dimension is read from the model, so comparing models means comparing
tables rather than fighting one fixed schema.
"""
import json, sys, unicodedata, psycopg2
from fastembed import TextEmbedding

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"

def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def prefix(model: str, text: str) -> str:
    """E5 models are trained with asymmetric prefixes; omitting them misrepresents the model."""
    return f"passage: {text}" if "e5" in model.lower() else text

def main():
    table = sys.argv[1] if len(sys.argv) > 1 else "items"
    model = sys.argv[2] if len(sys.argv) > 2 else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rows = json.load(open("out/catalogue.json"))
    emb = TextEmbedding(model_name=model)
    texts = [prefix(model, f"{r['name']}. {r['desc']}") for r in rows]
    vectors = list(emb.embed(texts))
    dim = len(vectors[0])
    print(f"model={model} dim={dim} table={table}")
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table};")
        cur.execute(f"""CREATE TABLE {table} (
            id int primary key, name text not null, descr text not null,
            search_text text not null, cat text, brand text, variant text,
            embedding vector({dim}) not null);""")
        cur.execute(f"CREATE INDEX {table}_tsv ON {table} USING gin (to_tsvector('simple', search_text));")
        cur.execute(f"""CREATE INDEX {table}_hnsw ON {table} USING hnsw (embedding vector_cosine_ops)
                       WITH (m = 16, ef_construction = 64);""")
        for r, v in zip(rows, vectors):
            cur.execute(f"INSERT INTO {table} (id,name,descr,search_text,cat,brand,variant,embedding)"
                        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                        (r["id"], r["name"], r["desc"],
                         strip_accents(f"{r['name']} {r['desc']}").lower(),
                         r["cat"], r["brand"], r["variant"],
                         "[" + ",".join(f"{x:.6f}" for x in v) + "]"))
        conn.commit()
        cur.execute(f"SELECT count(*), pg_size_pretty(pg_total_relation_size('{table}')), "
                    f"pg_size_pretty(pg_relation_size('{table}_hnsw')) FROM {table}")
        print("loaded", cur.fetchone())

if __name__ == "__main__":
    main()
