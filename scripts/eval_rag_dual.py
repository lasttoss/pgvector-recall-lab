#!/usr/bin/env python3
"""One document, two language versions, one embedding model.

The Pixon-shaped case: every knowledge source exists in Vietnamese and in English.
A chat message arrives in either language. Instead of picking one column, keep both
vectors on the same row and rank by the *best* distance across the two columns.
Same model for both columns, so the distances are comparable.
"""
import json, sys, psycopg2
from fastembed import TextEmbedding
sys.path.insert(0, "scripts")
from eval_translate import translate

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
MODEL = "intfloat/multilingual-e5-large"

def lit(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

def main():
    rows = json.load(open("out/catalogue.json"))
    qs = json.load(open("out/queries_paraphrase.json"))
    emb = TextEmbedding(model_name=MODEL)
    print("translating docs to english, embedding with e5-large …")
    docs_en = translate([f"{r['name']}. {r['desc']}" for r in rows])
    en_vecs = list(emb.embed([f"passage: {t}" for t in docs_en]))
    q_vi = list(emb.embed([f"query: {q['q']}" for q in qs]))
    q_en = list(emb.embed([f"query: {t}" for t in translate([q["q"] for q in qs])]))
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS items_en_e5;")
        cur.execute("CREATE TABLE items_en_e5 (id int primary key, embedding vector(1024) not null);")
        for r, v in zip(rows, en_vecs):
            cur.execute("INSERT INTO items_en_e5 VALUES (%s,%s)", (r["id"], lit(v)))
        conn.commit()

        def run(qv, both):
            vals = []
            for v, q in zip(qv, qs):
                if both:
                    cur.execute("""SELECT v.id, LEAST(v.embedding <=> %s::vector, e.embedding <=> %s::vector) d
                                   FROM items_e5large v JOIN items_en_e5 e USING (id)
                                   ORDER BY d LIMIT 10""", (lit(v), lit(v)))
                else:
                    cur.execute("SELECT id FROM items_en_e5 ORDER BY embedding <=> %s::vector LIMIT 10", (lit(v),))
                top = [r[0] for r in cur.fetchall()]
                gold = set(q["gold"])
                hit = [d for d in top if d in gold]
                vals.append((len(hit)/len(gold), next((1.0/i for i, d in enumerate(top,1) if d in gold), 0.0),
                             1.0 if hit else 0.0))
            n = len(vals)
            return tuple(sum(v[i] for v in vals)/n for i in range(3))

        vi_only = run(q_vi, False)                 # wrong on purpose: EN table, VI query (sanity)
        en_only = run(q_en, False)
        rowsout = [("VI query → VI column only (baseline)", None),
                   ("VI query → best of VI+EN per document", None),
                   ("EN query → EN column only", None),
                   ("EN query → best of VI+EN per document", None)]
        a = run(q_vi, True)
        b = run(q_en, True)
    # baseline VI column was measured in eval.py: recall .111 mrr .244 hit .625
    print(f"{'variant':44} {'recall@10':>10} {'MRR@10':>8} {'hit@10':>8}")
    print(f"{'VI query → VI column only (baseline)':44} {0.111:10.3f} {0.244:8.3f} {0.625:8.3f}")
    print(f"{'VI query → best of VI+EN per document':44} {a[0]:10.3f} {a[1]:8.3f} {a[2]:8.3f}")
    print(f"{'EN query → EN column only':44} {en_only[0]:10.3f} {en_only[1]:8.3f} {en_only[2]:8.3f}")
    print(f"{'EN query → best of VI+EN per document':44} {b[0]:10.3f} {b[1]:8.3f} {b[2]:8.3f}")
    print(f"{'VI query → EN column only (sanity: mismatched)':44} {vi_only[0]:10.3f} {vi_only[1]:8.3f} {vi_only[2]:8.3f}")

if __name__ == "__main__":
    main()
