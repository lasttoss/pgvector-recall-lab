#!/usr/bin/env python3
"""Translate Vietnamese to English, then embed with an English model.

The question this answers: is "translate first, search in English" a real alternative
to a multilingual embedding model? Both sides have to be translated — documents and
queries — otherwise the vectors live in different languages and nothing matches.
"""
import json, sys, unicodedata, psycopg2, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
sys.path.insert(0, "scripts")

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
MT = "Helsinki-NLP/opus-mt-vi-en"
EN_MODEL = "nomic-ai/nomic-embed-text-v1.5"

def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

_mt = {}

def translate(texts, batch=16):
    if not _mt:
        _mt["tok"] = AutoTokenizer.from_pretrained(MT)
        _mt["mdl"] = AutoModelForSeq2SeqLM.from_pretrained(MT).eval()
    tok, mdl = _mt["tok"], _mt["mdl"]
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=256)
            gen = mdl.generate(**enc, max_length=256, num_beams=1)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out

def embed_en(texts, batch=16):
    from fastembed import TextEmbedding
    return list(TextEmbedding(model_name=EN_MODEL).embed(texts))

def run(cur, table, qvecs, qs, vec_sql):
    res = []
    for v, q in zip(qvecs, qs):
        lit = "[" + ",".join(f"{x:.6f}" for x in v) + "]"
        cur.execute(f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT 10", (lit,))
        top = [r[0] for r in cur.fetchall()]
        gold = set(q["gold"])
        hit = [d for d in top if d in gold]
        mrr = next((1.0 / i for i, d in enumerate(top, 1) if d in gold), 0.0)
        res.append((len(hit) / len(gold), mrr, 1.0 if hit else 0.0))
    n = len(res)
    return (sum(r[0] for r in res)/n, sum(r[1] for r in res)/n, sum(r[2] for r in res)/n)

def main():
    rows = json.load(open("out/catalogue.json"))
    qs = json.load(open("out/queries_paraphrase.json"))
    print(f"translating {len(rows)} docs and {len(qs)} queries vi->en …")
    docs_en = translate([f"{r['name']}. {r['desc']}" for r in rows])
    qs_en = translate([q["q"] for q in qs])
    print("sample:", qs[0]["q"], "=>", qs_en[0])
    dvecs = embed_en([f"search_document: {t}" for t in docs_en])
    qvec = embed_en([f"search_query: {t}" for t in qs_en])
    qvec_vi = embed_en([f"search_query: {q['q']}" for q in qs])   # Vietnamese query, English docs
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS items_translated;")
        cur.execute("CREATE TABLE items_translated (id int primary key, embedding vector(768) not null);")
        for r, v in zip(rows, dvecs):
            cur.execute("INSERT INTO items_translated VALUES (%s,%s)",
                        (r["id"], "[" + ",".join(f"{x:.6f}" for x in v) + "]"))
        cur.execute("CREATE INDEX ON items_translated USING hnsw (embedding vector_cosine_ops);")
        conn.commit()
        a = run(cur, "items_translated", qvec, qs, None)
        b = run(cur, "items_translated", qvec_vi, qs, None)
    print(f"{'variant':46} {'recall@10':>10} {'mrr@10':>8} {'hit@10':>8}")
    print(f"{'translate docs + queries, English model':46} {a[0]:10.3f} {a[1]:8.3f} {a[2]:8.3f}")
    print(f"{'translate docs only, Vietnamese queries':46} {b[0]:10.3f} {b[1]:8.3f} {b[2]:8.3f}")

if __name__ == "__main__":
    main()
