"""Generate test predictions on all CPU cores.

    python predict_parallel.py --data-dir <test dir> --model <model.pkl> --out-dir <dir>
                               [--index-dir <dir>]

1. Builds the shared, memory-mapped index over test S2+S3 (skipped if present).
2. Scores every candidate pair of every test S1 in a process pool. Each finished chunk
   is written as a shard (all candidate ids + probabilities >= FLOOR + the sum of all
   probabilities), so an interrupted run resumes instead of restarting.
3. Finalizes over the WHOLE test set, which per-chunk processing cannot do:
   - per-entity expected-F0.5 selection (decide.select_expected_f)
   - one-owner rule: a target claimed by several S1 goes to the highest probability
4. Writes matching_results.tsv and candidate_pairs.tsv and checks coverage/subset rules.
"""
import argparse
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FLOOR = 0.02          # candidates below this can never be selected (decide.py floor)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tuples(path):
    import pandas as pd
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return list(zip(df["entity_id"], df["business_name"],
                    df["business_address"], df["country"]))


def _run_chunk(job):
    import pairs
    chunk_id, queries, shard_dir = job
    t0 = time.perf_counter()
    cand_lines, score_lines, n_pairs = [], [], 0
    for sid, tids, X, q, recs in pairs._iter_pairs(queries, with_records=True):
        cand_lines.append(f"{sid}\t{','.join(tids)}\n")
        if X is None:
            score_lines.append(f"{sid}\t0\t\n")
            continue
        p = pairs.score_query(q, tids, X, recs)       # stage 1 (+ stage 2 when in the model file)
        n_pairs += len(tids)
        kept = ",".join(f"{t}:{q:.5f}" for t, q in zip(tids, p.tolist()) if q >= FLOOR)
        score_lines.append(f"{sid}\t{float(p.sum()):.5f}\t{kept}\n")
    for kind, lines in (("cand", cand_lines), ("score", score_lines)):
        final = os.path.join(shard_dir, f"{kind}_{chunk_id:05d}.tsv")
        with open(final + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            f.writelines(lines)
        os.replace(final + ".tmp", final)
    open(os.path.join(shard_dir, f"done_{chunk_id:05d}"), "w").close()   # written last
    return chunk_id, len(queries), n_pairs, time.perf_counter() - t0


def finalize(s1_ids, n_chunks, shard_dir, out_dir):
    from decide import select_expected_f, resolve_ownership
    scores, pred = {}, {}
    for i in range(n_chunks):
        with open(os.path.join(shard_dir, f"score_{i:05d}.tsv"), encoding="utf-8") as f:
            for line in f:
                sid, total, kept = line.rstrip("\n").split("\t")
                c = [(x.rsplit(":", 1)[0], float(x.rsplit(":", 1)[1]))
                     for x in kept.split(",")] if kept else []
                scores[sid] = c
                pred[sid] = select_expected_f(c, total=float(total))
    n_before = sum(map(len, pred.values()))
    pred = resolve_ownership(pred, scores)
    n_after = sum(map(len, pred.values()))
    log(f"selected {n_before:,} links; one-owner rule removed {n_before - n_after:,}")

    expected = set(s1_ids)
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as out:
        out.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ids:
            out.write(f"{sid}\t{','.join(sorted(pred.get(sid, ())))}\n")
    seen, violations = set(), 0
    with open(os.path.join(out_dir, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="\n") as out:
        out.write("source1_entity_id\tcandidate_entity_ids\n")
        for i in range(n_chunks):
            with open(os.path.join(shard_dir, f"cand_{i:05d}.tsv"), encoding="utf-8") as f:
                for line in f:
                    sid, ids = line.rstrip("\n").split("\t")
                    if pred.get(sid) and not pred[sid] <= set(ids.split(",")):
                        violations += 1
                    seen.add(sid)
                    out.write(line)
    missing = expected - seen
    log(f"rows: {len(expected):,} S1, candidate rows {len(seen):,}, missing {len(missing)}, "
        f"matches outside candidates {violations}")
    if missing or violations or len(pred) != len(expected):
        raise SystemExit("output check failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder holding test_source*.tsv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()

    index_dir = args.index_dir or os.path.join(args.out_dir, "index")
    shard_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    if not any(os.path.isfile(os.path.join(index_dir, c, "meta.pkl"))
               for c in (os.listdir(index_dir) if os.path.isdir(index_dir) else [])):
        from shared_index import build
        log("building shared index over test S2+S3 ...")
        t0 = time.perf_counter()
        recs = (load_tuples(os.path.join(args.data_dir, "test_source2.tsv"))
                + load_tuples(os.path.join(args.data_dir, "test_source3.tsv")))
        build(recs, index_dir, log=log)
        del recs
        log(f"index built in {time.perf_counter() - t0:.0f}s")

    s1 = load_tuples(os.path.join(args.data_dir, "test_source1.tsv"))
    s1_ids = [r[0] for r in s1]
    s1.sort(key=lambda r: r[3])             # one country per chunk: cache locality
    chunks = [s1[i:i + args.chunk] for i in range(0, len(s1), args.chunk)]
    pending = [(i, c, shard_dir) for i, c in enumerate(chunks)
               if not os.path.exists(os.path.join(shard_dir, f"done_{i:05d}"))]
    log(f"S1={len(s1):,}  chunks={len(chunks)}  already done={len(chunks) - len(pending)}  "
        f"workers={args.workers}")

    if pending:
        import pairs
        t0 = time.perf_counter()
        done_q = done_pairs = 0
        todo_q = sum(len(j[1]) for j in pending)
        with pairs._pool(args.workers, index_dir, args.model) as pool:
            for cid, nq, npairs, secs in pool.imap_unordered(_run_chunk, pending):
                done_q += nq
                done_pairs += npairs
                el = time.perf_counter() - t0
                rate = done_q / el
                log(f"chunk {cid:05d} done  {done_q:,}/{todo_q:,} S1  {rate:,.0f} S1/s  "
                    f"{done_pairs / el:,.0f} pairs/s  ETA {(todo_q - done_q) / rate / 3600:.2f} h")

    log("finalizing ...")
    finalize(s1_ids, len(chunks), shard_dir, args.out_dir)
    log("done")


if __name__ == "__main__":
    main()
