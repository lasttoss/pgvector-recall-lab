# pgvector-recall-lab

Does hybrid retrieval actually beat lexical search on a Vietnamese product catalogue — and by how much?

A JD for an "AI Application / Backend Engineer" in Hanoi asks for PostgreSQL, Vector Search,
Search, Ranking and Recommendation in the same sentence. That implies `pgvector`, not
Elasticsearch. This lab measures whether that choice holds up, on data shaped like the
catalogue such a product would index: building materials, lighting, sanitary ware.

## What is measured

| Component | Choice |
|---|---|
| Catalogue | 900 synthetic items, 13 categories, 12 brands, Vietnamese names with diacritics |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2`, 384 dims, ONNX via `fastembed` (CPU only) |
| Lexical index | `tsvector('simple')` over unaccented text + GIN |
| Vector index | `pgvector` 0.8.7, HNSW (m=16, ef_construction=64), cosine |
| Fusion | Reciprocal Rank Fusion, k=60, depth 50 per retriever |
| Query set A | 60 queries built from item attributes; half typed **without diacritics**; gold = same category + brand + variant |
| Query set B | 40 **paraphrase** queries where no query word appears in the product name |

## Results — query set A (exact-ish, how a spec search looks)

| strategy | recall@10 | hit@10 | MRR@10 | p50 ms | empty queries |
|---|---|---|---|---|---|
| lexical | 1.000 | 1.000 | 1.000 | 0.30 | 0 |
| vector | 0.768 | 0.867 | 0.601 | 0.99 | 0 |
| hybrid (RRF) | 1.000 | 1.000 | 0.992 | 0.02 | 0 |

## Results — query set B (paraphrase, how a customer actually types)

| strategy | recall@10 | hit@10 | MRR@10 | p50 ms | empty queries |
|---|---|---|---|---|---|
| lexical | 0.000 | **0.000** | 0.000 | 0.18 | **40 / 40** |
| vector | 0.053 | 0.375 | 0.090 | 0.69 | 0 |
| hybrid (RRF) | 0.053 | 0.375 | 0.090 | 0.02 | 0 |

**The headline number is the 40 / 40.** Every paraphrase query — "đèn gắn trần ốp trần cho
phòng khách", "vòi nước cho chậu rửa mặt" — returns **zero rows** from BM25, because none of
those words appear in a product called "đèn LED âm trần Rạng Đông 9W 4000K". This is not a
tuning problem; it is what lexical search does.

**And the second headline: the default 384-dim model only found something relevant in 37.5% of
those queries** — see the model comparison below, where a stronger model reaches 0.625. A 384-dimension MiniLM is a weak retriever on a catalogue where 900 items differ
mostly by a wattage and a colour temperature. The bottleneck is the embedding model, not the
fusion: hybrid cannot beat the best of its inputs, and adding BM25's zeros changes nothing.


## Model comparison — the bottleneck is the model, not the fusion

Three embeddings, same catalogue, same queries, same database. `hit@10` is "did anything
relevant reach the page".

| model | dims | hit@10 paraphrase | hit@10 exact | recall@10 exact | encode docs/s (CPU) | HNSW bytes/row |
|---|---|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 0.375 | 0.867 | 0.768 | 88 | 1 966 |
| `minishlab/potion-multilingual-128M` (static) | 256 | 0.375 | 0.950 | 0.882 | **7 837** | 1 319 |
| `intfloat/multilingual-e5-large` | 1024 | **0.625** | **1.000** | **0.993** | 7 | 7 835 |

Three things this says:

1. **Upgrading the model moved paraphrase hit@10 from 0.375 to 0.625 — +25 points.** No amount
   of fusion tuning closes that gap, because RRF cannot create a relevant document the vector
   retriever never returned.
2. **The cost is not on the query path, it is on the ingest path.** e5-large encodes 7 docs/s on
   this CPU: re-embedding 100 000 products is ~4 hours, versus 13 seconds for the static model.
   That is the real argument in the meeting, not the benchmark table.
3. **A 256-dim static model beats a 384-dim transformer on exact queries** (0.950 vs 0.867) at
   90× the encode speed, and loses nothing on paraphrase (both 0.375). Cheap is not the same as
   worse; which one wins depends on the query mix, which is why the query sets are separated.

**Gotcha:** E5 models are trained with asymmetric prefixes. Without `query: ` and `passage: `
the numbers are wrong in a way that looks like "the model is bad". `scripts/build_index.py` and
`scripts/eval.py` apply them when the model name contains `e5`.

## ANN quality is not free, and at this size it is not even a speedup

Ground truth = exact top-10 over all rows with indexes disabled. Then the same queries through
HNSW at different `ef_search`:

| ef_search | recall vs exact | p50 ms | exact p50 ms |
|---|---|---|---|
| 10 | 0.975 | 0.23 | 0.64 |
| 40 | 1.000 | 0.62 | 0.62 |
| 100 | 1.000 | 0.59 | 0.58 |
| 200 | 1.000 | 0.57 | 0.55 |

At 900 rows a sequential scan is already sub-millisecond, so HNSW buys **nothing** here — the
numbers say so honestly instead of pretending the index earned its keep. What it does show is
the shape of the trade: `ef_search = 10` is ~2.5% off exact and that is the knob you turn,
not `m`.

## Storage, measured then extrapolated

```
900 rows · table 4336 kB · HNSW index 1728 kB · GIN index 568 kB
HNSW: 1966 bytes/row  (384 dims = 1536 B raw vectors + ~430 B graph)
⇒ 1 000 000 rows ≈ 1.9 GB just for the HNSW index, plus the table
```
The per-row number is measured; the 1M projection is arithmetic on top of it, not a benchmark.

## Reproduce

```bash
docker compose up -d
PYTHONPATH=/tmp/emblibs python3 scripts/gen_dataset.py      # 900 items
python3 scripts/queries.py && python3 scripts/queries_paraphrase.py
PYTHONPATH=/tmp/emblibs python3 scripts/build_index.py      # loads + embeds + indexes
PYTHONPATH=/tmp/emblibs python3 scripts/eval.py out/queries.json
PYTHONPATH=/tmp/emblibs python3 scripts/eval.py out/queries_paraphrase.json
PYTHONPATH=/tmp/emblibs python3 scripts/eval_ann.py
```

## What this does not prove

* **The data is synthetic.** Structure is realistic; the numbers say nothing about a real
  catalogue with photos, synonyms harvested from logs, and misspellings.
* **Relevance is defined by construction** (same category + brand + variant, or category +
  colour + colour temperature). No human judged these results. Real evaluation needs labelled
  queries, and that is the part everyone skips.
* **One embedding model.** The 0.375 hit@10 is a statement about a 384-dim MiniLM on this
  catalogue, not about vector search in general. A larger multilingual model would move it.
* **900 rows.** Index behaviour at 10^6 rows is a different regime; every latency number here
  is dominated by client overhead.

## What I would ask a team running this in production

1. How many items are in the index, and how often does the whole catalogue get re-embedded?
2. Which queries get labelled, by whom, and what does the regression test assert?
3. When a product's price changes but its text does not, what invalidates the embedding?
4. Has anyone compared hybrid against the plain lexical baseline, or is hybrid assumed better?
