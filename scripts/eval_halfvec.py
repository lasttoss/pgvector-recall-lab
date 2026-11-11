#!/usr/bin/env python3
"""What does fp16 (halfvec) cost in retrieval quality?

Same 900 documents, same model, same 250 queries — only the stored precision changes.
halfvec is 3x smaller on disk and 2.4x faster to query at 100k rows, so the question is
whether it costs anything that matters.
"""
import json, random, statistics
import psycopg2
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
MODEL, PREFIX = "intfloat/multilingual-e5-large", "query: "


def lit(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def boot(a, b, n=10000, seed=7):
    rnd, idx, d = random.Random(seed), range(len(a)), []
    for _ in range(n):
        s = [rnd.choice(idx) for _ in idx]
        d.append(sum(a[i] for i in s) / len(s) - sum(b[i] for i in s) / len(s))
    d.sort()
    return d[int(0.025 * n)], d[int(0.975 * n)]


def main():
    qs = json.load(open("out/queries_power.json"))
    slices = sorted({q["slice"] for q in qs})
    m = SentenceTransformer(MODEL, device="cpu")
    qv = m.encode([PREFIX + q["q"] for q in qs], normalize_embeddings=True,
                  batch_size=32, show_progress_bar=False)
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS zoo_e5_half;")
        cur.execute("CREATE TABLE zoo_e5_half (id int primary key, embedding halfvec(1024) not null);")
        cur.execute("INSERT INTO zoo_e5_half SELECT id, embedding::halfvec(1024) FROM zoo_e5_large__baseline;")
        cur.execute("CREATE INDEX ON zoo_e5_half USING hnsw (embedding halfvec_cosine_ops);")
        conn.commit()
        cur.execute("SELECT pg_relation_size('zoo_e5_half_embedding_idx'), "
                    "pg_relation_size('zoo_e5_large__baseline_embedding_idx')")
        half_ix, f32_ix = cur.fetchone()
        rows = []
        for v, q in zip(qv, qs):
            cur.execute("SELECT id FROM zoo_e5_half ORDER BY embedding <=> %s::halfvec LIMIT 10", (lit(v),))
            top = [r[0] for r in cur.fetchall()]
            gold = set(q["gold"])
            got = [d for d in top if d in gold]
            rows.append({"hit": 1.0 if got else 0.0, "recall": len(got) / len(gold),
                         "mrr": next((1.0 / i for i, d in enumerate(top, 1) if d in gold), 0.0)})
    base = json.load(open("out/power_results.json"))["scores"]["e5-large (baseline)"]
    hf_hit = [r["hit"] for r in rows]
    f32_hit = [r["hit"] for r in base]
    d = statistics.mean(hf_hit) - statistics.mean(f32_hit)
    lo, hi = boot(hf_hit, f32_hit)
    print(f"e5-large float32   hit@10={statistics.mean(f32_hit):.3f} "
          f"recall={statistics.mean(r['recall'] for r in base):.3f} "
          f"mrr={statistics.mean(r['mrr'] for r in base):.3f}")
    print(f"e5-large halfvec   hit@10={statistics.mean(hf_hit):.3f} "
          f"recall={statistics.mean(r['recall'] for r in rows):.3f} "
          f"mrr={statistics.mean(r['mrr'] for r in rows):.3f}")
    print(f"diff hit@10 = {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"-> {'KHÔNG mất gì đo được' if lo <= 0 <= hi else 'có khác biệt'}")
    print(f"index: float32 {f32_ix/1e6:.1f} MB vs halfvec {half_ix/1e6:.1f} MB "
          f"({f32_ix/max(half_ix,1):.1f}x nhỏ hơn, {f32_ix/900/1024:.1f} vs {half_ix/900/1024:.1f} KB/dòng)")
    print(f"\n{'slice':10} {'float32':>8} {'halfvec':>8}")
    for s in slices:
        a = [r["hit"] for r, q in zip(base, qs) if q["slice"] == s]
        b = [r["hit"] for r, q in zip(rows, qs) if q["slice"] == s]
        print(f"{s:10} {statistics.mean(a):8.3f} {statistics.mean(b):8.3f}")
    json.dump({"half": rows, "index_bytes": {"f32": f32_ix, "half": half_ix}},
              open("out/halfvec_results.json", "w"))


if __name__ == "__main__":
    main()
