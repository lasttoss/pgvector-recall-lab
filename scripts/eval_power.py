#!/usr/bin/env python3
"""Run every model against the 250-query set, per slice, with paired bootstrap.

40 queries could not separate these models. 250 can: SE(hit@10) drops from ~0.08 to
~0.03, and the paired bootstrap removes the variance shared between models.
"""
import json, random, statistics
import psycopg2
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
INSTRUCT = ("Instruct: Given a query about building materials and lighting products, "
            "retrieve product passages that answer it\nQuery: ")
MODELS = [
    ("e5-large (baseline)", "intfloat/multilingual-e5-large", "query: ", "zoo_e5_large__baseline"),
    ("bge-m3", "BAAI/bge-m3", "", "zoo_bge_m3"),
    ("Qwen3-0.6B + instruct", "Qwen/Qwen3-Embedding-0.6B", INSTRUCT, "zoo_qwen3_embedding_0_6b"),
    ("Vietnamese_Embedding (AITeamVN)", "AITeamVN/Vietnamese_Embedding", "",
     "zoo_vietnamese_embedding__aiteam"),
    ("vietnamese-bi-encoder (bkai)", "bkai-foundation-models/vietnamese-bi-encoder", "",
     "zoo_vietnamese_bi_encoder__bkai"),
]


def lit(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def score(cur, table, qv, qs):
    out = []
    for v, q in zip(qv, qs):
        cur.execute(f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT 10", (lit(v),))
        top = [r[0] for r in cur.fetchall()]
        gold = set(q["gold"])
        got = [d for d in top if d in gold]
        out.append({"hit": 1.0 if got else 0.0, "recall": len(got) / len(gold),
                    "mrr": next((1.0 / i for i, d in enumerate(top, 1) if d in gold), 0.0)})
    return out


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
    results = {}
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        for label, hf, pref, table in MODELS:
            m = SentenceTransformer(hf, device="cpu")
            qv = m.encode([pref + q["q"] for q in qs], normalize_embeddings=True,
                          batch_size=32, show_progress_bar=False)
            results[label] = score(cur, table, qv, qs)
            print(f"done {label}", flush=True)
            del m
    json.dump({"queries": [{"q": q["q"], "slice": q["slice"], "gold_size": q["gold_size"]} for q in qs],
               "scores": results}, open("out/power_results.json", "w"), ensure_ascii=False)

    base = [r["hit"] for r in results["e5-large (baseline)"]]
    print(f"\n{'model':34} {'hit@10':>7} {'recall':>7} {'MRR':>6} {'chênh e5':>9} {'95% CI':>17}  verdict")
    for label, rows in results.items():
        hit = [r["hit"] for r in rows]
        d = statistics.mean(hit) - statistics.mean(base)
        lo, hi = boot(hit, base)
        verdict = "không phân biệt" if lo <= 0 <= hi else ("TỐT HƠN" if d > 0 else "KÉM HƠN")
        print(f"{label:34} {statistics.mean(hit):7.3f} {statistics.mean(r['recall'] for r in rows):7.3f} "
              f"{statistics.mean(r['mrr'] for r in rows):6.3f} {d:+9.3f} [{lo:+.3f}, {hi:+.3f}]  {verdict}")

    print("\nper slice (hit@10):")
    print(f"{'model':34} " + " ".join(f"{s:>9}" for s in slices))
    for label, rows in results.items():
        cells = []
        for s in slices:
            vals = [r["hit"] for r, q in zip(rows, qs) if q["slice"] == s]
            cells.append(f"{statistics.mean(vals):9.3f}")
        print(f"{label:34} " + " ".join(cells))


if __name__ == "__main__":
    main()
