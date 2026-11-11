#!/usr/bin/env python3
"""How long does two-vector indexing take, and does the query get slow at scale?

Synthetic vectors (not real embeddings) so the numbers describe pgvector's index machinery,
not retrieval quality. Two tables of N x 1024 to model "accented" and "unaccented" columns.
"""
import io, json, os, statistics, sys, time
import numpy as np
import psycopg2

DSN = "postgresql://postgres:lab@127.0.0.1:55432/lab"
N = int(os.environ.get("N", 100_000))
DIM = 1024


def gen(n, rng):
    v = rng.standard_normal((n, DIM), dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def copy_in(cur, table, vecs):
    buf = io.StringIO()
    for i, row in enumerate(vecs, 1):
        buf.write(f"{i}\t[" + ",".join(f"{x:.5f}" for x in row) + "]\n")
    buf.seek(0)
    t0 = time.time()
    cur.copy_expert(f"COPY {table} (id, embedding) FROM STDIN", buf)
    return time.time() - t0


def main():
    rng = np.random.default_rng(42)
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '1GB'")
        cur.execute("SET max_parallel_maintenance_workers = 8")
        for t in ("scale_a", "scale_b"):
            cur.execute(f"DROP TABLE IF EXISTS {t}")
            cur.execute(f"CREATE TABLE {t} (id int primary key, embedding vector({DIM}) not null)")
        conn.commit()
        a = gen(N, rng)
        b = a + rng.standard_normal((N, DIM), dtype=np.float32) * 0.05   # near-duplicate: unaccented twin
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        out = {"n": N, "dim": DIM}
        out["copy_s_a"] = round(copy_in(cur, "scale_a", a), 1)
        out["copy_s_b"] = round(copy_in(cur, "scale_b", b), 1)
        conn.commit()
        print(f"loaded 2 x {N} rows in {out['copy_s_a']+out['copy_s_b']:.0f}s", flush=True)

        for t in ("scale_a", "scale_b"):
            cur.execute("SET maintenance_work_mem = '1GB'")
            t0 = time.time()
            cur.execute(f"CREATE INDEX ON {t} USING hnsw (embedding vector_cosine_ops)")
            conn.commit()
            out[f"build_s_{t}"] = round(time.time() - t0, 1)
            cur.execute(f"SELECT pg_relation_size('{t}'), pg_indexes_size('{t}')")
            tb, ix = cur.fetchone()
            out[f"table_mb_{t}"] = round(tb / 1e6)
            out[f"index_mb_{t}"] = round(ix / 1e6)
            print(f"{t}: build {out[f'build_s_{t}']}s · table {out[f'table_mb_{t}']}MB · index {out[f'index_mb_{t}']}MB", flush=True)

        qs = gen(200, np.random.default_rng(7))

        def lit(v):
            return "[" + ",".join(f"{x:.5f}" for x in v) + "]"

        def timeit(sql_fn):
            lat = []
            for v in qs:
                t0 = time.time()
                sql_fn(lit(v))
                lat.append((time.time() - t0) * 1000)
            lat.sort()
            return round(statistics.median(lat), 2), round(lat[int(0.95 * len(lat))], 2)

        results = {}
        for ef in (40, 100):
            cur.execute(f"SET hnsw.ef_search = {ef}")
            cur.execute("SET hnsw.iterative_scan = relaxed_order")

            def one(x):
                cur.execute("SELECT id FROM scale_a ORDER BY embedding <=> %s::vector LIMIT 10", (x,))
                cur.fetchall()

            def both(x):
                cur.execute("""SELECT a.id, LEAST(a.embedding <=> %s::vector, b.embedding <=> %s::vector) d
                               FROM scale_a a JOIN scale_b b USING (id) ORDER BY d LIMIT 10""", (x, x))
                cur.fetchall()

            results[f"single_ef{ef}"] = timeit(one)
            results[f"dual_least_ef{ef}"] = timeit(both)
            print(f"ef={ef}: single {results[f'single_ef{ef}']} ms · dual {results[f'dual_least_ef{ef}']} ms", flush=True)
        out["latency_p50_p95"] = results
        json.dump(out, open(f"out/scale_{N}.json", "w"), indent=1)
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
