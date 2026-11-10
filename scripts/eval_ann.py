#!/usr/bin/env python3
"""HNSW is approximate: measure what ef_search buys and what it costs.

Ground truth = exact top-10 by scanning every row (index disabled). Then the same
query through HNSW at several ef_search values, comparing overlap with exact.
"""
import json, statistics, time, psycopg2
from fastembed import TextEmbedding

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def main():
    qs = json.load(open("out/queries.json"))[:40]
    emb = TextEmbedding(model_name=MODEL)
    vecs = list(emb.embed([q["q"] for q in qs]))
    rows = []
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        for ef in (10, 40, 100, 200):
            cur.execute(f"SET hnsw.ef_search = {ef}")
            overlaps, lat = [], []
            exact_lat = []
            for v in vecs:
                lit = "[" + ",".join(f"{x:.6f}" for x in v) + "]"
                t0 = time.perf_counter()
                cur.execute("SELECT id FROM items ORDER BY embedding <=> %s::vector LIMIT 10", (lit,))
                ann = [r[0] for r in cur.fetchall()]
                lat.append(time.perf_counter() - t0)
                cur.execute("SET LOCAL enable_indexscan = off; SET LOCAL enable_bitmapscan = off")
                t0 = time.perf_counter()
                cur.execute("SELECT id FROM items ORDER BY embedding <=> %s::vector LIMIT 10", (lit,))
                exact = set(r[0] for r in cur.fetchall())
                exact_lat.append(time.perf_counter() - t0)
                cur.execute("RESET enable_indexscan; RESET enable_bitmapscan")
                overlaps.append(len(set(ann) & exact) / min(10, len(exact)))
            rows.append((ef, statistics.mean(overlaps), statistics.median(lat) * 1000,
                         statistics.median(exact_lat) * 1000))
    print(f"{'ef_search':>9} {'recall vs exact':>16} {'p50 ms':>8} {'exact p50 ms':>13}")
    for ef, rec, p50, exact in rows:
        print(f"{ef:9d} {rec:16.3f} {p50:8.2f} {exact:13.2f}")
    json.dump([{"ef_search": ef, "recall_vs_exact": round(rec, 4), "p50_ms": round(p50, 2),
                "exact_p50_ms": round(exact, 2)} for ef, rec, p50, exact in rows],
              open("out/results_ann.json", "w"), indent=2)

if __name__ == "__main__":
    main()
