#!/usr/bin/env python3
"""Model shootout for Vietnamese retrieval, run through one harness.

Every model sees the same 900 catalogue rows and both query sets, so the numbers
are comparable. Pooling and prompts come from each model's own sentence-transformers
config rather than hand-rolled assumptions, and one broken model does not stop the run.
"""
import json, os, statistics, sys, time
import psycopg2
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
OUT = "out/model_zoo.json"

SPECS = [
    ("e5-large (baseline)", "intfloat/multilingual-e5-large", "query: ", "passage: ", False),
    ("gte-multilingual-base", "Alibaba-NLP/gte-multilingual-base", "", "", False),
    ("embeddinggemma-300m (Google)", "google/embeddinggemma-300m", "", "", False),
    ("bge-m3", "BAAI/bge-m3", "", "", False),
    ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B", "", "", False),
    ("Vietnamese_Embedding (AITeamVN)", "AITeamVN/Vietnamese_Embedding", "", "", False),
    ("vietnamese-bi-encoder (bkai)", "bkai-foundation-models/vietnamese-bi-encoder", "", "", False),
]


def lit(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def metrics(cur, table, qvecs, qs):
    rec = []
    for v, q in zip(qvecs, qs):
        cur.execute(f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT 10", (lit(v),))
        top = [r[0] for r in cur.fetchall()]
        gold = set(q["gold"])
        hit = [d for d in top if d in gold]
        rec.append((len(hit) / len(gold),
                    next((1.0 / i for i, d in enumerate(top, 1) if d in gold), 0.0),
                    1.0 if hit else 0.0))
    n = len(rec)
    return tuple(round(sum(r[i] for r in rec) / n, 3) for i in range(3))


def run_one(model, label, qp, dp, rows, docs, sets):
    dim = int(model.get_embedding_dimension())
    slug = "zoo_" + "".join(c if c.isalnum() else "_" for c in label.lower())[:28].strip("_")
    prompts = set(getattr(model, "prompts", None) or {})

    def enc(texts, kind):
        if kind in prompts:
            return model.encode(texts, prompt_name=kind, normalize_embeddings=True,
                                batch_size=16, show_progress_bar=False)
        pref = qp if kind == "query" else dp
        return model.encode([pref + t for t in texts], normalize_embeddings=True,
                            batch_size=16, show_progress_bar=False)

    t0 = time.time()
    dvecs = enc(docs, "document")
    entry = {"hf": label, "dim": dim, "docs_per_s": round(len(docs) / (time.time() - t0), 1),
             "prompts": sorted(prompts)}
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {slug};")
        cur.execute(f"CREATE TABLE {slug} (id int primary key, embedding vector({dim}) not null);")
        for r, v in zip(rows, dvecs):
            cur.execute(f"INSERT INTO {slug} VALUES (%s,%s)", (r["id"], lit(v)))
        cur.execute(f"CREATE INDEX ON {slug} USING hnsw (embedding vector_cosine_ops);")
        cur.execute(f"SELECT pg_relation_size('{slug}_embedding_idx')")
        entry["index_bytes_per_row"] = round(cur.fetchone()[0] / len(rows))
        conn.commit()
        for name, qs in sets.items():
            t0 = time.time()
            qvecs = enc([q["q"] for q in qs], "query")
            enc_q = (time.time() - t0) / len(qs) * 1000
            lat = []
            for v in qvecs[:15]:
                t0 = time.time()
                cur.execute(f"SELECT id FROM {slug} ORDER BY embedding <=> %s::vector LIMIT 10", (lit(v),))
                cur.fetchall()
                lat.append((time.time() - t0) * 1000)
            entry[name] = {"metrics": metrics(cur, slug, qvecs, qs),
                           "query_ms": round(statistics.median(lat), 2),
                           "encode_query_ms": round(enc_q, 1)}
    return entry


def main():
    rows = json.load(open("out/catalogue.json"))
    sets = {"paraphrase": json.load(open("out/queries_paraphrase.json")),
            "spec": json.load(open("out/queries.json"))}
    docs = [f"{r['name']}. {r['desc']}" for r in rows]
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for label, hf, qp, dp, trust in SPECS:
        if label in results:
            print("skip (done):", label, flush=True)
            continue
        try:
            model = SentenceTransformer(hf, device="cpu", trust_remote_code=trust)
        except Exception as e:
            print(f"LOAD FAIL {label}: {type(e).__name__}: {e}"[:300], flush=True)
            continue
        try:
            entry = run_one(model, label, qp, dp, rows, docs, sets)
        except Exception as e:
            print(f"RUN FAIL {label}: {type(e).__name__}: {e}"[:300], flush=True)
            del model
            continue
        results[label] = entry
        json.dump(results, open(OUT, "w"), indent=1, ensure_ascii=False)
        print(f"{label:34} dim={entry['dim']:5} docs/s={entry['docs_per_s']:6.1f} "
              f"para={entry['paraphrase']['metrics']} spec={entry['spec']['metrics']}", flush=True)
        del model
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
