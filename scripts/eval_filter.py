#!/usr/bin/env python3
"""Filtered vector search: the one place pgvector quietly loses recall.

HNSW walks the graph and only *then* applies the WHERE clause, so a selective
filter can yield fewer than k rows or the wrong rows. pgvector 0.8 added
iterative index scans; this measures whether that fixes it on our data.
"""
import json, statistics, sys, time, psycopg2
from fastembed import TextEmbedding

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
sys.path.insert(0, "scripts")

def main():
    table, model = "items_e5large", "intfloat/multilingual-e5-large"
    rows = json.load(open("out/catalogue.json"))
    cat_counts = {}
    for r in rows:
        cat_counts[r["cat"]] = cat_counts.get(r["cat"], 0) + 1
    small = sorted(cat_counts.items(), key=lambda kv: kv[1])[:3]
    print("smallest categories:", small)
    cat = small[0][0]
    qs = json.load(open("out/queries_paraphrase.json"))[:40]
    emb = TextEmbedding(model_name=model)
    vecs = list(emb.embed([f"query: {q['q']}" for q in qs]))
    out = {}
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        for label, setup in (("hnsw plain", []),
                             ("hnsw iterative relaxed", ["SET hnsw.iterative_scan = relaxed_order"]),
                             ("exact (no index)", ["SET LOCAL enable_indexscan = off"])):
            for s in setup:
                cur.execute(s)
            cur.execute("SET hnsw.ef_search = 40")
            rows_out, lat = [], []
            for v in vecs:
                lit = "[" + ",".join(f"{x:.6f}" for x in v) + "]"
                t0 = time.perf_counter()
                cur.execute(f"SELECT id FROM {table} WHERE cat = %s ORDER BY embedding <=> %s::vector LIMIT 10",
                            (cat, lit))
                rows_out.append(len(cur.fetchall()))
                lat.append((time.perf_counter() - t0) * 1000)
            out[label] = (statistics.mean(rows_out), min(rows_out), statistics.median(lat))
            cur.execute("RESET hnsw.iterative_scan; RESET enable_indexscan")
    print(f"filter cat='{cat}' ({cat_counts[cat]} rows of 900)")
    print(f"{'mode':24} {'rows returned':>14} {'min':>5} {'p50 ms':>8}")
    for k, (avg, mn, p50) in out.items():
        print(f"{k:24} {avg:14.2f} {mn:5d} {p50:8.2f}")

if __name__ == "__main__":
    main()
