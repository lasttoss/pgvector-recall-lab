#!/usr/bin/env python3
"""Paired bootstrap over queries: is a model really better, or is 40 queries too few?

With 40 queries a 0.05 difference in hit@10 is well inside sampling noise, so a
leaderboard without a confidence interval invites the wrong conclusion.
"""
import json, random, statistics
import psycopg2
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
INSTRUCT = ("Instruct: Given a query about building materials and lighting products, "
            "retrieve product passages that answer it\nQuery: ")
MODELS = [
    ("e5-large (baseline)", "intfloat/multilingual-e5-large", "query: ", "zoo_e5_large__baseline", None),
    ("bge-m3", "BAAI/bge-m3", "", "zoo_bge_m3", None),
    ("Qwen3-0.6B + instruct", "Qwen/Qwen3-Embedding-0.6B", INSTRUCT, "zoo_qwen3_embedding_0_6b", None),
    ("Vietnamese_Embedding (AITeamVN)", "AITeamVN/Vietnamese_Embedding", "",
     "zoo_vietnamese_embedding__aiteam", None),
    ("vietnamese-bi-encoder (bkai)", "bkai-foundation-models/vietnamese-bi-encoder", "",
     "zoo_vietnamese_bi_encoder__bkai", None),
]


def lit(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def per_query(cur, table, qv, qs):
    hits, recs = [], []
    for v, q in zip(qv, qs):
        cur.execute(f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT 10", (lit(v),))
        top = [r[0] for r in cur.fetchall()]
        gold = set(q["gold"])
        got = [d for d in top if d in gold]
        hits.append(1.0 if got else 0.0)
        recs.append(len(got) / len(gold))
    return hits, recs


def boot(a, b, n=10000, seed=7):
    rnd = random.Random(seed)
    idx = range(len(a))
    diffs = []
    for _ in range(n):
        s = [rnd.choice(idx) for _ in idx]
        diffs.append(sum(a[i] for i in s) / len(s) - sum(b[i] for i in s) / len(s))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def main():
    qs = json.load(open("out/queries_paraphrase.json"))
    data = {}
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        for label, hf, pref, table, _ in MODELS:
            m = SentenceTransformer(hf, device="cpu")
            qv = m.encode([pref + q["q"] for q in qs], normalize_embeddings=True,
                          batch_size=16, show_progress_bar=False)
            data[label] = per_query(cur, table, qv, qs)
            print(f"{label:34} hit@10={statistics.mean(data[label][0]):.3f}", flush=True)
            del m
    base = data["e5-large (baseline)"][0]
    print(f"\n{'model':34} {'hit@10':>7} {'diff vs e5':>11} {'95% CI':>18}  verdict")
    for label, hit in data.items():
        d = statistics.mean(hit[0]) - statistics.mean(base)
        lo, hi = boot(hit[0], base)
        verdict = "không phân biệt được" if lo <= 0 <= hi else ("tốt hơn" if d > 0 else "kém hơn")
        print(f"{label:34} {statistics.mean(hit[0]):7.3f} {d:+11.3f} [{lo:+.3f}, {hi:+.3f}]  {verdict}")


if __name__ == "__main__":
    main()
