# Product search for a Design & Build catalogue — nine decisions, each with a number

Every claim below comes from a run in this repository against 900 Vietnamese SKUs across 13
categories (lighting, sanitary ware, tiles and stone, industrial wood, switches and sockets) and a
250-query benchmark whose gold sets are computed from catalogue attributes, never from a model.
Reproduce with `make zip`-free targets: `make data && make index && make power && make bootstrap`.

## 1. Keyword search alone is not enough, and vector alone is not enough

| query set | lexical recall@10 | lexical MRR@10 | vector recall@10 | vector MRR@10 |
|---|---|---|---|---|
| specification ("đèn led âm trần 9w 4000k") | 1.000 | 1.000 | 0.984 | 0.917 |
| paraphrase ("đèn gắn trần phòng khách") | **0.000** | **0.000** | 0.106 | 0.251 |

Specification queries are 100% solvable by `tsvector` and an exact match. Paraphrase queries share
no tokens with the product name, so BM25 returns **nothing at all** for 40 out of 40. Run both.

## 2. Hybrid, not replacement

Hybrid (lexical + vector, fused) scored 1.000 recall / 1.000 MRR on the specification set and never
scored below the vector-only result on paraphrase. There is no measured case in this catalogue where
adding lexical hurt. Cost: one extra index, one extra query.

## 3. The embedding model is not the lever

250 queries, paired bootstrap against `multilingual-e5-large`:

| model | hit@10 | diff | 95% CI | verdict |
|---|---|---|---|---|
| `multilingual-e5-large` | 0.696 | — | — | baseline |
| `BAAI/bge-m3` | 0.676 | -0.020 | [-0.064, +0.020] | tie |
| `AITeamVN/Vietnamese_Embedding` | 0.676 | -0.020 | [-0.060, +0.020] | tie |
| `bkai/vietnamese-bi-encoder` | 0.620 | -0.076 | [-0.132, -0.024] | worse |
| `Qwen3-Embedding-0.6B` | 0.588 | -0.108 | [-0.164, -0.052] | worse |

At 40 queries every interval contained zero, so the earlier ranking was noise. At 250 two models are
confidently worse and two are ties. **Choose on cost, latency and licence, not on a leaderboard.**

## 4. Stripping Vietnamese diacritics halves every model

| model | plain | verbose | nodauc | brand |
|---|---|---|---|---|
| e5-large | 0.762 | 0.790 | **0.302** | 0.935 |
| bge-m3 | 0.794 | 0.774 | 0.238 | 0.903 |
| Vietnamese_Embedding | 0.730 | 0.758 | 0.333 | 0.887 |
| bkai bi-encoder | 0.746 | 0.548 | 0.270 | 0.919 |
| Qwen3 + instruct | 0.651 | 0.694 | 0.159 | 0.855 |

Filler words cost nothing (0.790 vs 0.762). Missing diacritics cost more than half the accuracy, for
every model, and that is how a large share of phone input arrives.

## 5. Fix diacritics with a second vector, not with a bigger model

Store the document twice — accented and unaccented — with the same model, and rank by `LEAST`:

| variant | hit@10 on 63 unaccented queries | diff | 95% CI |
|---|---|---|---|
| accented index only | 0.302 | — | — |
| unaccented index only | 0.413 | +0.111 | [-0.016, +0.238] |
| best of both | **0.429** | **+0.127** | **[+0.016, +0.254]** |

Cost: one more encode per document, one more index (7.9 KB/row at 1024 dims), ~2 ms per query. The
query side needs no language or diacritic detection.

## 6. A second ranking stage buys more than any first-stage model

| setup | hit@10 | MRR@10 | latency/query |
|---|---|---|---|
| vector only | 0.625 | 0.244 | 38 ms |
| vector + cross-encoder-style rerank of the top candidates | **0.725** | **0.344** | 774 ms |

+0.10 hit for +736 ms. That belongs in an "ask AI" box, not in a search-as-you-type field.

## 7. Do not translate the catalogue into English

| pipeline | hit@10 | MRR@10 | added latency |
|---|---|---|---|
| Vietnamese corpus + multilingual model | 0.625 | 0.244 | 0 |
| translate documents only, Vietnamese queries | 0.150 | 0.047 | — |
| translate documents + queries, English model | 0.550 | 0.290 | 205 ms/query |

Translating one side is a silent catastrophe. Translating both is a wash on quality and a
permanent cost; it is justified when the *reader* is English-speaking, not when the *index* is.

## 8. Two language versions of one document: one row, two vectors, best distance

Same model in both columns, `LEAST(dist_vi, dist_en)`: Vietnamese queries reproduced 0.625 and
English queries 0.650, exactly the single-column scores, with no language detector. Forcing a
Vietnamese query onto an English column costs 0.625 -> 0.500, which is the number to quote when
someone proposes "just index the English version".

## 9. What I would do first, before choosing anything

1. **Log the query mix** for two weeks: specification vs natural language, accented vs unaccented,
   with-brand vs without. Every decision above is weighted by that mix, and it is different for
   contractors (spec-heavy) and homeowners (natural-language-heavy).
2. **Build a 250-query gold set** with gold derived from attributes. Without it, no change can be
   shown to work: the 40-query set could not distinguish five models from each other.
3. **Then** pick the model on cost, add the unaccented vector, and put reranking behind the slow
   entry point only.

## What this does not prove

- 900 synthetic SKUs with one-sentence descriptions: no long-tail attributes, no typos, no
  multi-attribute specs (`1200x600x12mm`, IP rating, CRI), no supplier-specific part numbers.
- 250 queries is enough to separate 0.08 differences, not 0.02 ones. Two ties remain ties.
- Filtered ANN search did not reproduce the known recall degradation at this corpus size, so this
  repository cannot say whether pgvector holds up at 10^5-10^6 rows with selective filters.
- Latency figures are CPU-only on one machine, with a warm index and no concurrent writers.
