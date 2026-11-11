#!/usr/bin/env python3
"""Diacritics are the real Vietnamese weakness, not the model.

Every model loses more than half its hit@10 when the query drops diacritics — which is
how a large share of real phone input arrives. This tests the cheap fix: index a second
vector of the same document with diacritics stripped, and rank by the best of the two.
"""
import json, random, statistics, unicodedata, time
import psycopg2
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
MODEL = "intfloat/multilingual-e5-large"


def strip_accents(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


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
    rows = json.load(open("out/catalogue.json"))
    qs = [q for q in json.load(open("out/queries_power.json")) if q["slice"] == "nodauc"]
    m = SentenceTransformer(MODEL, device="cpu")
    t0 = time.time()
    dvecs = m.encode([f"passage: {strip_accents(r['name'])}. {strip_accents(r['desc'])}" for r in rows],
                     normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    print(f"stripped docs encoded: {len(rows)/(time.time()-t0):.1f} docs/s", flush=True)
    qvecs = m.encode([f"query: {q['q']}" for q in qs], normalize_embeddings=True,
                     batch_size=32, show_progress_bar=False)
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS zoo_e5_nodauc;")
        cur.execute("CREATE TABLE zoo_e5_nodauc (id int primary key, embedding vector(1024) not null);")
        for r, v in zip(rows, dvecs):
            cur.execute("INSERT INTO zoo_e5_nodauc VALUES (%s,%s)", (r["id"], lit(v)))
        cur.execute("CREATE INDEX ON zoo_e5_nodauc USING hnsw (embedding vector_cosine_ops);")
        conn.commit()
        hits = {"accented index only": [], "stripped index only": [], "best of both (LEAST)": []}
        for v, q in zip(qvecs, qs):
            gold = set(q["gold"])
            cur.execute("SELECT id FROM zoo_e5_large__baseline ORDER BY embedding <=> %s::vector LIMIT 10", (lit(v),))
            a = [r[0] for r in cur.fetchall()]
            cur.execute("SELECT id FROM zoo_e5_nodauc ORDER BY embedding <=> %s::vector LIMIT 10", (lit(v),))
            b = [r[0] for r in cur.fetchall()]
            cur.execute("""SELECT v.id, LEAST(v.embedding <=> %s::vector, s.embedding <=> %s::vector) d
                           FROM zoo_e5_large__baseline v JOIN zoo_e5_nodauc s USING (id)
                           ORDER BY d LIMIT 10""", (lit(v), lit(v)))
            c = [r[0] for r in cur.fetchall()]
            for name, top in (("accented index only", a), ("stripped index only", b), ("best of both (LEAST)", c)):
                hits[name].append(1.0 if any(d in gold for d in top) else 0.0)
    base = hits["accented index only"]
    print(f"\n{'variant':26} {'hit@10':>7} {'chênh':>7} {'95% CI':>17}")
    for name, h in hits.items():
        d = statistics.mean(h) - statistics.mean(base)
        lo, hi = boot(h, base)
        print(f"{name:26} {statistics.mean(h):7.3f} {d:+7.3f} [{lo:+.3f}, {hi:+.3f}]")
    json.dump({k: round(statistics.mean(v), 4) for k, v in hits.items()},
              open("out/diacritics_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
