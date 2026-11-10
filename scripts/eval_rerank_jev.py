#!/usr/bin/env python3
"""Rerank the vector recall list with a decision model (typesafe/jev) and compare.

Architecture under test: cheap local recall (256-dim static model) + API rerank.
This is the retrieve-then-rerank shape that lets a CPU-only box serve good search.

The API is called through curl because the endpoint rejects non-browser TLS
fingerprints (Cloudflare 1010), which is worth knowing before you promise
a "simple HTTP client" in a design doc.
"""
import json, statistics, subprocess, sys, time, psycopg2
from fastembed import TextEmbedding

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
TABLE = sys.argv[2] if len(sys.argv) > 2 else "items_potion"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "minishlab/potion-multilingual-128M"
URL = "https://api.commandcode.ai/provider/v1/systemone"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
CAND = 20

def jev(key, state, questions):
    body = json.dumps({"model": "typesafe/jev", "state": state, "questions": questions})
    out = subprocess.run(["curl", "-s", "--max-time", "60", "-X", "POST", URL,
                          "-H", f"Authorization: Bearer {key}", "-H", "Content-Type: application/json",
                          "-A", UA, "-d", body], capture_output=True, text=True).stdout
    return json.loads(out)

def main():
    key = json.load(open("/home/dy52337/.pi/agent/auth.json"))["commandcode"]["key"]
    qfile = sys.argv[1] if len(sys.argv) > 1 else "out/queries_paraphrase.json"
    qs = json.load(open(qfile))
    def prefix(t):
        m = MODEL.lower()
        if "e5" in m:
            return f"query: {t}"
        if "nomic" in m:
            return f"search_query: {t}"
        return t
    emb = TextEmbedding(model_name=MODEL)
    qvecs = list(emb.embed([prefix(q["q"]) for q in qs]))
    rows = []
    tokens = 0
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        for q, vec in zip(qs, qvecs):
            lit = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
            cur.execute(f"SELECT id, name, descr FROM {TABLE} ORDER BY embedding <=> %s::vector LIMIT %s",
                        (lit, CAND))
            cands = cur.fetchall()
            state = (f"Customer search query: {q['q']}\nCandidates:\n"
                     + "\n".join(f"{i}. {n} — {d}" for i, (_, n, d) in enumerate(cands, 1)))
            questions = {f"c{i}": {"type": "noul",
                                   "instructions": f"Is candidate {i} a relevant match for the customer's query?"}
                         for i in range(1, len(cands) + 1)}
            t0 = time.perf_counter()
            resp = jev(key, state, questions)
            ms = (time.perf_counter() - t0) * 1000
            tokens += resp.get("usage", {}).get("input_tokens", 0)
            probs = [(cands[i - 1][0], resp["answers"][f"c{i}"]["noul"]) for i in range(1, len(cands) + 1)]
            ranked = [doc for doc, _ in sorted(probs, key=lambda kv: -kv[1])]
            rows.append({"q": q["q"], "gold": set(q["gold"]), "ranked": ranked, "ms": ms})
    def metrics(ranked, gold, k=10):
        top = ranked[:k]
        hit = [d for d in top if d in gold]
        mrr = next((1.0 / i for i, d in enumerate(top, 1) if d in gold), 0.0)
        return len(hit) / len(gold), mrr, (1.0 if hit else 0.0)
    rec, mrr, hit, lat = [], [], [], []
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        for r in rows:
            a, b, c = metrics(r["ranked"], r["gold"])
            rec.append(a); mrr.append(b); hit.append(c); lat.append(r["ms"])
    print(f"== JEV rerank over {TABLE} recall, {len(rows)} queries, {CAND} candidates")
    print(f"recall@10 {statistics.mean(rec):.3f} · hit@10 {statistics.mean(hit):.3f} · "
          f"MRR@10 {statistics.mean(mrr):.3f} · p50 {statistics.median(lat):.0f} ms · "
          f"input tokens {tokens} (${tokens/1e6*0.04:.4f})")
    json.dump({"recall@10": round(statistics.mean(rec), 4), "hit@10": round(statistics.mean(hit), 4),
               "mrr@10": round(statistics.mean(mrr), 4), "p50_ms": round(statistics.median(lat), 1),
               "input_tokens": tokens, "cost_usd": round(tokens/1e6*0.04, 5),
               "queries": [{"q": r["q"], "ms": round(r["ms"], 1)} for r in rows]},
              open("out/rerank_jev.json", "w"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
