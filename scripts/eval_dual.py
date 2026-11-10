#!/usr/bin/env python3
"""Two columns, two indexes: Vietnamese original + English translation, searched together.

This is the "store both at ingest" design. The question is whether fusing a Vietnamese
index and an English index beats one multilingual index over the Vietnamese text.
"""
import json, statistics, sys, psycopg2, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
sys.path.insert(0, "scripts")
from fastembed import TextEmbedding

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
MT = "Helsinki-NLP/opus-mt-vi-en"
_mt = {}

def translate(texts, batch=16):
    if not _mt:
        _mt["tok"] = AutoTokenizer.from_pretrained(MT)
        _mt["mdl"] = AutoModelForSeq2SeqLM.from_pretrained(MT).eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            enc = _mt["tok"](texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=256)
            out.extend(_mt["tok"].batch_decode(_mt["mdl"].generate(**enc, max_length=256, num_beams=1),
                                               skip_special_tokens=True))
    return out

def rrf(*lists, k=60):
    score = {}
    for lst in lists:
        for rank, doc in enumerate(lst, 1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(score.items(), key=lambda kv: -kv[1])]

def search(cur, table, lit, depth=50):
    cur.execute(f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s", (lit, depth))
    return [r[0] for r in cur.fetchall()]

def metrics(ranked, gold, k=10):
    top = ranked[:k]
    hit = [d for d in top if d in gold]
    mrr = next((1.0 / i for i, d in enumerate(top, 1) if d in gold), 0.0)
    return len(hit) / len(gold), mrr, (1.0 if hit else 0.0)

def main():
    qs = json.load(open("out/queries_paraphrase.json"))
    vi = TextEmbedding(model_name="intfloat/multilingual-e5-large")
    en = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
    qvi = list(vi.embed([f"query: {q['q']}" for q in qs]))
    qen = en.embed([f"search_query: {t}" for t in translate([q["q"] for q in qs])])
    qen = list(qen)
    variants = {"vi only (e5-large)": [], "en only (translated)": [], "vi + en fused (RRF)": []}
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        for a, b, q in zip(qvi, qen, qs):
            la = "[" + ",".join(f"{x:.6f}" for x in a) + "]"
            lb = "[" + ",".join(f"{x:.6f}" for x in b) + "]"
            r_vi = search(cur, "items_e5large", la)
            r_en = search(cur, "items_translated", lb)
            gold = set(q["gold"])
            for name, ranked in (("vi only (e5-large)", r_vi), ("en only (translated)", r_en),
                                 ("vi + en fused (RRF)", rrf(r_vi, r_en))):
                variants[name].append(metrics(ranked, gold))
    print(f"{'variant':26} {'recall@10':>10} {'MRR@10':>8} {'hit@10':>8}")
    for name, vals in variants.items():
        n = len(vals)
        print(f"{name:26} {sum(v[0] for v in vals)/n:10.3f} {sum(v[1] for v in vals)/n:8.3f} {sum(v[2] for v in vals)/n:8.3f}")

if __name__ == "__main__":
    main()
