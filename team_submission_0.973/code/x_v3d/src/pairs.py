"""Candidate generation + featurization + scoring, parallel across all cores.

One code path serves training-pair generation, dev/holdout scoring and test inference,
so features are computed identically everywhere. Workers memory-map the shared index;
BLAS/OpenMP threads are pinned to 1 per process to avoid 24x24 oversubscription.

v3b: scoring is two-stage. score_query() applies the pair model (stage 1), then, when the model file
carries a stage-2 model, re-scores candidates with p >= stage2.FLOOR from their list context
(stage2.py). The same function serves dev scoring, test inference and stage-2 training data.
"""
import os
import pickle
import time

import numpy as np

_W = {}


def _init(index_dir, model_path, s1_counts=None):
    from shared_index import SharedBlocker
    _W["blocker"] = SharedBlocker(index_dir)
    _W["s1cnt"] = {}
    if s1_counts:
        with open(s1_counts, "rb") as f:
            _W["s1cnt"] = pickle.load(f)
    _W["model"] = _W["stage2"] = None
    if model_path:
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        _W["model"] = obj["model"]
        _W["stage2"] = obj.get("stage2")


def _iter_pairs(queries, batch=100, with_records=False):
    """Yields (s1_id, target_ids, X[, query, records]) per query; X is None for zero candidates."""
    from features import QueryInfo, featurize
    from norm import norm_name, norm_addr, base_norm
    from shared_index import INDIC
    blocker = _W["blocker"]
    for s in range(0, len(queries), batch):
        qs = queries[s:s + batch]
        qn = [(norm_name(q[1]) or base_norm(q[1]).strip(), norm_addr(q[2], q[3])) for q in qs]
        for q, (nn, na), res in zip(qs, qn, blocker.candidates_batch(qs, qn)):
            if res is None or not len(res[1]):
                yield (q[0], [], None, q, []) if with_records else (q[0], [], None)
                continue
            ci, cl, nc, ac, rb, wc = res
            qi = QueryInfo(q, nn, na, ci, bool(INDIC.search(q[1])))
            m = len(cl)
            rank_n = np.argsort(np.argsort(-nc, kind="stable"), kind="stable")
            rank_a = np.argsort(np.argsort(-ac, kind="stable"), kind="stable")
            mx = float(nc.max())
            n_close = int((nc >= 0.9 * mx).sum()) if mx > 0 else 0
            tids, rows, recs = [], [], []
            for j in range(m):
                t = ci.record(int(cl[j]))
                tids.append(t[0])
                recs.append(t)
                rows.append(featurize(qi, t, float(nc[j]), float(ac[j]), int(rb[j]),
                                      int(rank_n[j]), int(rank_a[j]), m, n_close, float(wc[j])))
            X = np.asarray(rows, dtype=np.float32)
            yield (q[0], tids, X, q, recs) if with_records else (q[0], tids, X)


def s1_name_count(country, nn):
    """How many S1 records of this split and country have this normalized name (sorted tokens)."""
    from shared_index import name_key
    return _W["s1cnt"].get(country + "|" + name_key(nn), 0) if nn else 0


def context_of(q, tids, p1, recs, X):
    """Stage-2 inputs for candidates with p1 >= FLOOR: (indices, context rows)."""
    from stage2 import FLOOR, S1_FEATS, context_rows
    from norm import norm_name, base_norm
    idx = [j for j in range(len(tids)) if p1[j] >= FLOOR]
    cands = [(tids[j], float(p1[j]), recs[j][2], recs[j][3], recs[j][4]) for j in idx]
    qn = norm_name(q[1]) or base_norm(q[1]).strip()
    qc = s1_name_count(q[3], qn)
    extra = []
    for j in idx:
        cc = s1_name_count(q[3], recs[j][3])
        same = 1 if (recs[j][3] and recs[j][3].split() and
                     sorted(set(recs[j][3].split())) == sorted(set(qn.split()))) else 0
        extra.append([float(qc), float(cc), float(max(cc - same, 0))] + [float(X[j, c]) for c in S1_FEATS])
    return idx, context_rows(q[2], cands, extra)


def score_query(q, tids, X, recs):
    """Final probabilities for one S1: stage 1, then stage 2 when available."""
    p = _W["model"].score(X)
    s2 = _W["stage2"]
    if s2 is not None:
        idx, C = context_of(q, tids, p, recs, X)
        if idx:
            p = p.copy()
            p[idx] = s2.predict_proba(C)[:, 1]
    return p


def _gen_job(job):
    """Training pairs: write one .npz shard of features + ids."""
    shard_path, queries = job
    s1s, tids, xs = [], [], []
    for sid, t, X in _iter_pairs(queries):
        if X is not None:
            s1s.extend([sid] * len(t))
            tids.extend(t)
            xs.append(X)
    np.savez(shard_path, X=np.concatenate(xs) if xs else np.zeros((0, 1), np.float32),
             s1=np.array(s1s), t=np.array(tids))
    return len(queries), len(tids)


def _ctx_job(job):
    """Stage-2 training rows: stage-1 scores (out of sample) -> context features, one .npz shard."""
    shard_path, queries = job
    s1s, tids, cs = [], [], []
    model = _W["model"]
    for sid, t, X, q, recs in _iter_pairs(queries, with_records=True):
        if X is None:
            continue
        idx, C = context_of(q, t, model.score(X), recs, X)
        if idx:
            s1s.extend([sid] * len(idx))
            tids.extend(t[j] for j in idx)
            cs.append(C)
    np.savez(shard_path, C=np.concatenate(cs) if cs else np.zeros((0, 1), np.float32),
             s1=np.array(s1s), t=np.array(tids))
    return len(queries), len(tids)


def _score_job(queries):
    """Scoring: {s1_id: [(target_id, prob), ...]} for every candidate."""
    out = {}
    for sid, t, X, q, recs in _iter_pairs(queries, with_records=True):
        out[sid] = list(zip(t, score_query(q, t, X, recs).tolist())) if X is not None else []
    return out


def _pool(workers, index_dir, model_path):
    import multiprocessing as mp
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    s1c = os.path.join(index_dir, "s1_namecount.pkl")
    return mp.Pool(workers, initializer=_init,
                   initargs=(index_dir, model_path, s1c if os.path.isfile(s1c) else None))


def generate(queries, index_dir, shard_dir, workers=None, chunk=1000, log=print, model_path=None):
    """Featurize all candidate pairs of `queries` into shards (or stage-2 context rows when a
    model_path is given); returns shard paths."""
    os.makedirs(shard_dir, exist_ok=True)
    queries = sorted(queries, key=lambda r: r[3])
    prefix = "ctx" if model_path else "pairs"
    jobs = [(os.path.join(shard_dir, f"{prefix}_{i:05d}.npz"), queries[s:s + chunk])
            for i, s in enumerate(range(0, len(queries), chunk))]
    t0 = time.perf_counter()
    nq = npairs = 0
    with _pool(workers or os.cpu_count(), index_dir, model_path) as pool:
        for q, p in pool.imap_unordered(_ctx_job if model_path else _gen_job, jobs):
            nq += q
            npairs += p
    dt = time.perf_counter() - t0
    log(f"generated {npairs:,} {prefix} rows for {nq:,} S1 in {dt:.0f}s "
        f"({npairs / dt:,.0f} rows/s, {nq / dt:,.0f} S1/s)")
    return [j[0] for j in jobs]


def score(queries, index_dir, model_path, workers=None, chunk=500, log=print):
    queries = sorted(queries, key=lambda r: r[3])
    jobs = [queries[s:s + chunk] for s in range(0, len(queries), chunk)]
    t0 = time.perf_counter()
    out = {}
    with _pool(workers or os.cpu_count(), index_dir, model_path) as pool:
        for part in pool.imap_unordered(_score_job, jobs):
            out.update(part)
    dt = time.perf_counter() - t0
    npairs = sum(len(v) for v in out.values())
    log(f"scored {npairs:,} pairs for {len(out):,} S1 in {dt:.0f}s "
        f"({npairs / dt:,.0f} pairs/s, {len(out) / dt:,.0f} S1/s)")
    return out
