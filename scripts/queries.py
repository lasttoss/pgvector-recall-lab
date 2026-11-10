#!/usr/bin/env python3
"""Build evaluation queries from the catalogue, with relevance defined by construction.

relevance = same category AND same brand AND same variant/size attribute, so a
query has 3-8 relevant items rather than exactly one. Half the queries are typed
without diacritics, which is how Vietnamese users actually search.
"""
import json, random, unicodedata

def strip_accents(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def main(n_queries=60, seed=11):
    rows = json.load(open("out/catalogue.json"))
    rnd = random.Random(seed)
    picks = rnd.sample(rows, n_queries)
    queries = []
    for r in picks:
        q = f"{r['name'].split(' ' + r['brand'])[0]} {r['brand']} {r['variant']}"
        if rnd.random() < 0.5:
            q = strip_accents(q)
        if rnd.random() < 0.25:  # customer-style shorthand
            q = q.replace("đèn LED ", "den ").replace("đèn ", "den ")
        gold = [x["id"] for x in rows
                if x["cat"] == r["cat"] and x["brand"] == r["brand"] and x["variant"] == r["variant"]]
        queries.append({"q": q, "gold": gold})
    json.dump(queries, open("out/queries.json", "w"), ensure_ascii=False, indent=0)
    print(f"wrote {len(queries)} queries, avg gold size {sum(len(q['gold']) for q in queries)/len(queries):.1f}")

if __name__ == "__main__":
    main()
