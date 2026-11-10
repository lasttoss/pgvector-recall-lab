#!/usr/bin/env python3
"""Compare lexical, vector and hybrid retrieval on the same queries and table.

usage: eval.py <query-file> [table] [model]
"""
import json, statistics, sys, time, unicodedata, psycopg2
sys.path.insert(0, "scripts")
from fastembed import TextEmbedding

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
K, DEPTH, RRF_K = 10, 50, 60

def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def prefix(model: str, text: str) -> str:
    m = model.lower()
    if "e5" in m:
        return f"query: {text}"
    if "nomic" in m:
        return f"search_query: {text}"
    return text

def lexical(cur, table, q):
    a = strip_accents(q).lower()
    cur.execute(f"""SELECT id FROM {table}
        WHERE to_tsvector('simple', search_text) @@ websearch_to_tsquery('simple', %s)
        ORDER BY ts_rank_cd(to_tsvector('simple', search_text),
                            websearch_to_tsquery('simple', %s)) DESC LIMIT %s""", (a, a, DEPTH))
    return [r[0] for r in cur.fetchall()]

def vector(cur, table, vec):
    cur.execute(f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s",
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
    mrr = 0.0
    for i, d in enumerate(top, start=1):
        if d in gold:
            mrr = 1.0 / i
            break
    return len(hit) / len(gold), mrr, (1.0 if hit else 0.0)

def main():
    qfile = sys.argv[1] if len(sys.argv) > 1 else "out/queries.json"
    table = sys.argv[2] if len(sys.argv) > 2 else "items"
    model = sys.argv[3] if len(sys.argv) > 3 else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    qs = json.load(open(qfile))
    qtexts = [prefix(model, q["q"]) for q in qs]
    if "nomic-embed-text-v2" in model:
        import nomicv2
        qvecs = nomicv2.embed(qtexts)
    else:
        qvecs = list(TextEmbedding(model_name=model).embed(qtexts))
    res = {k: [[], [], []] for k in ("lexical", "vector", "hybrid")}
    lat = {k: [] for k in res}
    empty = {k: 0 for k in res}
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = 100")
        for q, vec in zip(qs, qvecs):
            gold = set(q["gold"])
            t0 = time.perf_counter(); lex = lexical(cur, table, q["q"]); lat["lexical"].append(time.perf_counter()-t0)
            t0 = time.perf_counter(); vec_ids = vector(cur, table, vec); lat["vector"].append(time.perf_counter()-t0)
            t0 = time.perf_counter(); hyb = rrf(lex, vec_ids); lat["hybrid"].append(time.perf_counter()-t0)
            for name, ranked in (("lexical", lex), ("vector", vec_ids), ("hybrid", hyb)):
                r, m, h = metrics(ranked, gold)
                res[name][0].append(r); res[name][1].append(m); res[name][2].append(h)
                if not ranked:
                    empty[name] += 1
    print(f"== {qfile} · table={table} · model={model.split('/')[-1]}")
    print(f"{'strategy':10} {'recall@10':>10} {'hit@10':>8} {'mrr@10':>8} {'p50 ms':>8} {'empty':>6}")
    for name in ("lexical", "vector", "hybrid"):
        rec = statistics.mean(res[name][0]); hit = statistics.mean(res[name][2])
        mrr = statistics.mean(res[name][1]); p50 = statistics.median(lat[name]) * 1000
        print(f"{name:10} {rec:10.3f} {hit:8.3f} {mrr:8.3f} {p50:8.2f} {empty[name]:6d}")

if __name__ == "__main__":
    main()
