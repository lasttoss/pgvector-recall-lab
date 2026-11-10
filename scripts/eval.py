#!/usr/bin/env python3
"""Compare lexical, vector and hybrid retrieval on the same queries.

recall@10 answers "did we find the relevant items at all"; MRR@10 answers "did we
put one at the top". Both matter, and they move differently.
"""
import json, statistics, time, unicodedata, psycopg2
from fastembed import TextEmbedding

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
K, DEPTH, RRF_K = 10, 50, 60

def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def lexical(cur, q):
    cur.execute("""SELECT id FROM items
                   WHERE to_tsvector('simple', search_text) @@ websearch_to_tsquery('simple', %s)
                   ORDER BY ts_rank_cd(to_tsvector('simple', search_text),
                                       websearch_to_tsquery('simple', %s)) DESC
                   LIMIT %s""", (strip_accents(q).lower(), strip_accents(q).lower(), DEPTH))
    return [r[0] for r in cur.fetchall()]

def vector(cur, vec):
    cur.execute("SELECT id FROM items ORDER BY embedding <=> %s::vector LIMIT %s",
                ("[" + ",".join(f"{x:.6f}" for x in vec) + "]", DEPTH))
    return [r[0] for r in cur.fetchall()]

def rrf(*lists):
    score = {}
    for lst in lists:
        for rank, doc in enumerate(lst, start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (RRF_K + rank)
    return [d for d, _ in sorted(score.items(), key=lambda kv: -kv[1])]

def metrics(ranked, gold):
    top = ranked[:K]
    hit = [d for d in top if d in gold]
    recall = len(hit) / len(gold)
    mrr = 0.0
    for i, d in enumerate(top, start=1):
        if d in gold:
            mrr = 1.0 / i
            break
    hit10 = 1.0 if hit else 0.0   # did anything relevant make the page at all
    return recall, mrr, hit10

def main():
    import sys
    qfile = sys.argv[1] if len(sys.argv) > 1 else "out/queries.json"
    qs = json.load(open(qfile))
    print(f"== {qfile} ({len(qs)} queries)")
    emb = TextEmbedding(model_name=MODEL)
    qvecs = list(emb.embed([q["q"] for q in qs]))
    results = {k: [[], [], []] for k in ("lexical", "vector", "hybrid")}
    lat = {"lexical": [], "vector": [], "hybrid": []}
    empty = {"lexical": 0, "vector": 0, "hybrid": 0}
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = 100")
        for q, vec in zip(qs, qvecs):
            gold = set(q["gold"])
            t0 = time.perf_counter(); lex = lexical(cur, q["q"]); lat["lexical"].append(time.perf_counter()-t0)
            t0 = time.perf_counter(); vec_ids = vector(cur, vec); lat["vector"].append(time.perf_counter()-t0)
            t0 = time.perf_counter(); hyb = rrf(lex, vec_ids); lat["hybrid"].append(time.perf_counter()-t0)
            for name, ranked in (("lexical", lex), ("vector", vec_ids), ("hybrid", hyb)):
                r, m, h = metrics(ranked, gold)
                results[name][0].append(r); results[name][1].append(m); results[name][2].append(h)
                if not ranked:
                    empty[name] += 1
    print(f"{'strategy':10} {'recall@10':>10} {'hit@10':>8} {'mrr@10':>8} {'p50 ms':>8} {'empty q':>8}")
    out = {}
    for name in ("lexical", "vector", "hybrid"):
        rec = statistics.mean(results[name][0]); mrr = statistics.mean(results[name][1])
        p50 = statistics.median(lat[name]) * 1000
        hit = statistics.mean(results[name][2])
        print(f"{name:10} {rec:10.3f} {hit:8.3f} {mrr:8.3f} {p50:8.2f} {empty[name]:8d}")
        out[name] = {"recall@10": round(rec, 4), "hit@10": round(hit, 4), "mrr@10": round(mrr, 4),
                     "p50_ms": round(p50, 2), "empty_queries": empty[name]}
    json.dump(out, open("out/results.json", "w"), indent=2)

if __name__ == "__main__":
    main()
