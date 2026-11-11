#!/usr/bin/env python3
"""Guess the diacritics back from context, instead of paying for a second index.

A single token is ambiguous by construction: "may" is may/may/may/may. An n-gram is not,
because the neighbouring words pick one reading. So this builds a lexicon from the
catalogue text itself (names, descriptions, categories, attributes), matches at n-gram
length 3 then 2 then 1, and measures four arms on the same 63 unaccented queries:

  1. no restoration, one accented index      - the baseline this repo already reported
  2. two indexes, best-of-both (UNION ALL)   - the fix in eval_diacritics.py
  3. phrase restoration, one accented index  - restoring the query first
  4. restoration + two indexes               - do they stack?

Plus a safety arm: restoring text that already has correct diacritics must not break it.
"""
import json, random, re, statistics, time, unicodedata
from collections import Counter

import psycopg2
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
MODEL = "intfloat/multilingual-e5-large"
MAXN = 3


def strip_accents(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def lit(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def boot(a, b, n=10000, seed=7):
    rnd, idx = random.Random(seed), range(len(a))
    d = sorted(sum(a[i] for i in [rnd.choice(idx) for _ in idx]) / len(a)
               - sum(b[i] for i in [rnd.choice(idx) for _ in idx]) / len(b) for _ in range(n))
    return d[int(0.025 * n)], d[int(0.975 * n)]


def build_lexicon(rows):
    """stripped n-gram -> Counter of accented n-grams, from catalogue text only."""
    counts = {n: {} for n in range(1, MAXN + 1)}
    fields = []
    for r in rows:
        for k in ("name", "desc", "cat", "brand", "variant", "size", "color"):
            if r.get(k):
                fields.append(str(r[k]))
    for text in fields:
        toks = [t.lower() for t in re.findall(r"\w+", text, flags=re.UNICODE)]
        for n in range(1, MAXN + 1):
            for i in range(len(toks) - n + 1):
                key = " ".join(strip_accents(t) for t in toks[i:i + n])
                counts[n].setdefault(key, Counter())[" ".join(toks[i:i + n])] += 1
    return counts


def restore(text, counts, only_missing=True):
    """Greedy longest n-gram match, one restored word per input word.

    only_missing=True keeps any word that already carries diacritics, which is what a
    mixed input needs: half the sentence typed with tone marks, half without."""
    toks = re.findall(r"\w+|[^\w\s]+", text, flags=re.UNICODE)
    is_word = [bool(re.match(r"^\w+$", t, flags=re.UNICODE)) for t in toks]
    low = [t.lower() for t, w in zip(toks, is_word) if w]
    mark = [strip_accents(t) != t for t in low]          # already carries diacritics
    out, i = [], 0
    while i < len(low):
        if only_missing and mark[i]:
            out.append(low[i])
            i += 1
            continue
        matched = None
        for n in range(min(MAXN, len(low) - i), 0, -1):
            if only_missing and any(mark[i:i + n]):
                continue
            c = counts[n].get(" ".join(strip_accents(t) for t in low[i:i + n]))
            if c:
                matched = c.most_common(1)[0][0].split()
                break
        out.extend(matched or [low[i]])
        i += len(matched) if matched else 1
    res, k = [], 0
    for t, w in zip(toks, is_word):
        if w:
            res.append(out[k])
            k += 1
        elif t in ",.;:!?":
            res[-1] += t
        else:
            res.append(t)
    return " ".join(res).strip()


def accented_originals():
    """Pair each unaccented query with the accented string it was stripped from.

    Re-runs the generator with strip_accents wrapped; the regenerated query file is
    byte-identical to the shipped one, which is asserted by the caller's md5 check."""
    import sys
    sys.path.insert(0, "scripts")
    import gen_queries_power as G
    cap, orig = {}, G.strip_accents

    def patched(s):
        out = orig(s)
        cap[out] = s
        return out

    G.strip_accents = patched
    G.main(target=250, seed=11)
    return cap


def top10(cur, v):
    cur.execute("SELECT id FROM zoo_e5_large__baseline ORDER BY embedding <=> %s::vector LIMIT 10", (lit(v),))
    return [r[0] for r in cur.fetchall()]


def main():
    rows = json.load(open("out/catalogue.json"))
    all_q = json.load(open("out/queries_power.json"))
    nod = [q for q in all_q if q["slice"] == "nodauc"]
    plain = [q for q in all_q if q["slice"] == "plain"]

    pairs = accented_originals()
    paired = [(q["q"], pairs[q["q"]]) for q in nod if q["q"] in pairs]
    print(f"unaccented queries {len(nod)} · paired with their accented original {len(paired)}", flush=True)

    counts = build_lexicon(rows)
    print(f"lexicon from catalogue text: {len(counts[1])} unigrams · {len(counts[2])} bigrams · "
          f"{len(counts[3])} trigrams", flush=True)

    tok_total = tok_exact = tok_restored = whole = 0
    for nod_s, acc_s in paired:
        got = restore(nod_s, counts)
        whole += (got == acc_s)
        for a, b in zip(re.findall(r"\w+", got), re.findall(r"\w+", acc_s)):
            tok_total += 1
            tok_exact += (a == b)
            tok_restored += (strip_accents(a) != a)
    print(f"restoration: {tok_exact}/{tok_total} tokens exact ({tok_exact / tok_total:.1%}), "
          f"{whole}/{len(paired)} whole queries exact", flush=True)
    for nod_s, acc_s in paired[:3]:
        print(f"  in   {nod_s}\n  out  {restore(nod_s, counts)}\n  gốc  {acc_s}", flush=True)

    m = SentenceTransformer(MODEL, device="cpu")
    restored = [restore(q["q"], counts) for q in nod]
    v_none = m.encode([f"query: {q['q']}" for q in nod], normalize_embeddings=True, batch_size=32)
    v_rest = m.encode([f"query: {x}" for x in restored], normalize_embeddings=True, batch_size=32)
    v_plain = m.encode([f"query: {q['q']}" for q in plain], normalize_embeddings=True, batch_size=32)
    # blind re-derivation (never looking at whether the word already has marks) is the
    # arm that shows why the guard matters, so keep it explicit
    v_plain_r = m.encode([f"query: {restore(q['q'], counts, only_missing=False)}" for q in plain],
                         normalize_embeddings=True, batch_size=32)

    arms = {k: [] for k in ("no restoration", "two indexes (UNION ALL)", "restoration, one index",
                            "restoration + two indexes")}
    t_sum = {k: 0.0 for k in ("single", "merged")}
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        for v, vr, q in zip(v_none, v_rest, nod):
            gold = set(q["gold"])
            arms["no restoration"].append(1.0 if any(d in gold for d in top10(cur, v)) else 0.0)
            t0 = time.perf_counter()
            arms["restoration, one index"].append(1.0 if any(d in gold for d in top10(cur, vr)) else 0.0)
            t_sum["single"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            cur.execute("""WITH cand AS (
                             (SELECT id, embedding <=> %s::vector AS d FROM zoo_e5_large__baseline ORDER BY 2 LIMIT 50)
                             UNION ALL
                             (SELECT id, embedding <=> %s::vector AS d FROM zoo_e5_nodauc ORDER BY 2 LIMIT 50))
                           SELECT id FROM cand GROUP BY id ORDER BY MIN(d) LIMIT 10""", (lit(v), lit(v)))
            arms["two indexes (UNION ALL)"].append(1.0 if any(d in gold for d in [r[0] for r in cur.fetchall()]) else 0.0)
            t_sum["merged"] += time.perf_counter() - t0
            cur.execute("""WITH cand AS (
                             (SELECT id, embedding <=> %s::vector AS d FROM zoo_e5_large__baseline ORDER BY 2 LIMIT 50)
                             UNION ALL
                             (SELECT id, embedding <=> %s::vector AS d FROM zoo_e5_nodauc ORDER BY 2 LIMIT 50))
                           SELECT id FROM cand GROUP BY id ORDER BY MIN(d) LIMIT 10""", (lit(vr), lit(vr)))
            arms["restoration + two indexes"].append(1.0 if any(d in gold for d in [r[0] for r in cur.fetchall()]) else 0.0)

        safety = {"typed with diacritics": [], "restored (all words)": [],
                  "restored (only missing)": []}
        for v, vr, q in zip(v_plain, v_plain_r, plain):
            gold = set(q["gold"])
            safety["typed with diacritics"].append(1.0 if any(d in gold for d in top10(cur, v)) else 0.0)
            safety["restored (all words)"].append(1.0 if any(d in gold for d in top10(cur, vr)) else 0.0)
            vm = m.encode([f"query: {restore(q['q'], counts, only_missing=True)}"],
                          normalize_embeddings=True)[0]
            safety["restored (only missing)"].append(1.0 if any(d in gold for d in top10(cur, vm)) else 0.0)

    base = arms["no restoration"]
    print(f"\n{'arm (63 unaccented queries)':0} ", end="")
    print(f"{'hit@10':>7} {'chênh':>7} {'95% CI vs no restoration':>26}")
    for k, h in arms.items():
        d = statistics.mean(h) - statistics.mean(base)
        lo, hi = boot(h, base)
        print(f"{k:32} {statistics.mean(h):7.3f} {d:+7.3f} [{lo:+.3f}, {hi:+.3f}]")
    lo, hi = boot(arms["restoration, one index"], arms["two indexes (UNION ALL)"])
    print(f"\nrestoration vs the second index: "
          f"{statistics.mean(arms['restoration, one index']) - statistics.mean(arms['two indexes (UNION ALL)']):+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}]")

    b = safety["typed with diacritics"]
    print(f"\nsafety on already-accented input (63 queries, {statistics.mean(b):.3f} baseline)")
    for k in ("restored (all words)", "restored (only missing)"):
        lo, hi = boot(safety[k], b)
        print(f"  {k:26} {statistics.mean(safety[k]):.3f} "
              f"({statistics.mean(safety[k]) - statistics.mean(b):+.3f} [{lo:+.3f}, {hi:+.3f}])")

    # mixed input: the realistic phone case, half the words typed without marks
    rnd = random.Random(23)
    mixed = []
    for q in plain:
        words = q["q"].split()
        keep = [w if rnd.random() < 0.5 else strip_accents(w) for w in words]
        mixed.append(" ".join(keep))
    vm = m.encode([f"query: {t}" for t in mixed], normalize_embeddings=True, batch_size=32)
    vr = m.encode([f"query: {restore(t, counts)}" for t in mixed], normalize_embeddings=True, batch_size=32)
    res = {"mixed, no restoration": [], "mixed, restored": []}
    with psycopg2.connect(DSN) as conn2, conn2.cursor() as cur:
        for v, vr2, q in zip(vm, vr, plain):
            gold = set(q["gold"])
            res["mixed, no restoration"].append(1.0 if any(d in gold for d in top10(cur, v)) else 0.0)
            res["mixed, restored"].append(1.0 if any(d in gold for d in top10(cur, vr2)) else 0.0)
    lo, hi = boot(res["mixed, restored"], res["mixed, no restoration"])
    print(f"\nmixed input, half the words stripped (63 queries)")
    for k, v in res.items():
        print(f"  {k:26} {statistics.mean(v):.3f}")
    print(f"  restoration gain           {statistics.mean(res['mixed, restored']) - statistics.mean(res['mixed, no restoration']):+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}]")
    print(f"query time at 900 rows: single index {t_sum['single'] / len(nod) * 1000:.2f} ms · "
          f"two indexes merged {t_sum['merged'] / len(nod) * 1000:.2f} ms")

    json.dump({"arms": {k: round(statistics.mean(v), 4) for k, v in arms.items()},
               "safety": {k: round(statistics.mean(v), 4) for k, v in safety.items()},
               "mixed": {k: round(statistics.mean(v), 4) for k, v in res.items()},
               "tokens_exact": tok_exact, "tokens_total": tok_total, "queries_exact": whole,
               "ms_single": round(t_sum["single"] / len(nod) * 1000, 2),
               "ms_merged": round(t_sum["merged"] / len(nod) * 1000, 2)},
              open("out/diacritics_restore.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
