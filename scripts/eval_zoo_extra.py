#!/usr/bin/env python3
"""Two fixes the shootout needs: gte with CLS pooling done by hand (its remote code
crashes under sentence-transformers), and Qwen3 with the instruction prefix it was
trained with. Qwen3 only needs queries re-encoded, so its document table is reused."""
import json, statistics, time
import psycopg2, torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
INSTRUCT = ("Instruct: Given a query about building materials and lighting products, "
            "retrieve product passages that answer it\nQuery: ")


def lit(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def metrics(cur, table, qvecs, qs):
    per = []
    for v, q in zip(qvecs, qs):
        cur.execute(f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT 10", (lit(v),))
        top = [r[0] for r in cur.fetchall()]
        gold = set(q["gold"])
        hit = [d for d in top if d in gold]
        per.append((len(hit) / len(gold), next((1.0 / i for i, d in enumerate(top, 1) if d in gold), 0.0),
                    1.0 if hit else 0.0))
    n = len(per)
    return tuple(round(sum(p[i] for p in per) / n, 3) for i in range(3))


def main():
    rows = json.load(open("out/catalogue.json"))
    sets = {"paraphrase": json.load(open("out/queries_paraphrase.json")),
            "spec": json.load(open("out/queries.json"))}
    docs = [f"{r['name']}. {r['desc']}" for r in rows]
    out = {}

    # ── gte-multilingual-base: CLS pooling, no prefixes ──
    name = "Alibaba-NLP/gte-multilingual-base"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(name, trust_remote_code=True).eval()

    def gte(texts, batch=16):
        vecs = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch):
                enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=512,
                          return_tensors="pt")
                h = mdl(**enc).last_hidden_state[:, 0]          # CLS
                vecs.extend(torch.nn.functional.normalize(h, p=2, dim=1).float().numpy())
        return vecs

    t0 = time.time()
    dvecs = gte(docs)
    out["gte-multilingual-base"] = {"dim": len(dvecs[0]), "docs_per_s": round(len(docs) / (time.time() - t0), 1)}
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS zoo_gte_multilingual_base;")
        cur.execute(f"CREATE TABLE zoo_gte_multilingual_base (id int primary key, embedding vector({len(dvecs[0])}) not null);")
        for r, v in zip(rows, dvecs):
            cur.execute("INSERT INTO zoo_gte_multilingual_base VALUES (%s,%s)", (r["id"], lit(v)))
        cur.execute("CREATE INDEX ON zoo_gte_multilingual_base USING hnsw (embedding vector_cosine_ops);")
        cur.execute("SELECT pg_relation_size('zoo_gte_multilingual_base_embedding_idx')")
        out["gte-multilingual-base"]["index_bytes_per_row"] = round(cur.fetchone()[0] / len(rows))
        conn.commit()
        for k, qs in sets.items():
            t0 = time.time()
            qv = gte([q["q"] for q in qs])
            out["gte-multilingual-base"][k] = {"metrics": metrics(cur, "zoo_gte_multilingual_base", qv, qs),
                                              "encode_query_ms": round((time.time() - t0) / len(qs) * 1000, 1)}
        print("gte-multilingual-base", out["gte-multilingual-base"], flush=True)

    # ── Qwen3 with the instruction prefix it was trained with ──
    q = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    table = [t for t in ("zoo_qwen3_embedding_0_6b",) if t]
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        for k, qs in sets.items():
            t0 = time.time()
            qv = q.encode([INSTRUCT + x["q"] for x in qs], normalize_embeddings=True,
                          batch_size=16, show_progress_bar=False)
            out.setdefault("Qwen3-Embedding-0.6B + instruct", {})[k] = {
                "metrics": metrics(cur, table[0], qv, qs),
                "encode_query_ms": round((time.time() - t0) / len(qs) * 1000, 1)}
        print("Qwen3 + instruct", out["Qwen3-Embedding-0.6B + instruct"], flush=True)
    json.dump(out, open("out/model_zoo_extra.json", "w"), indent=1, ensure_ascii=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
