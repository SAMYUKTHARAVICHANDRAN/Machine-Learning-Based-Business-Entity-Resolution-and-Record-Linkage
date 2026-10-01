"""Learn word-evidence tables for features.py from labelled candidate lists (fit split only).

For S1 entities of the fit split that the matcher does NOT train on, every scored candidate
(p >= 0.02 in a previous model's forward run) is compared with the S1 name: tokens the candidate adds
and tokens it drops (typos within edit distance 1 ignored). Per token,

    evidence(w) = log P(w | false candidate) - log P(w | true link)      (add-one smoothed)

kept when the token occurs at least MIN_COUNT times; clipped to [-4, 8]. Positive = sibling marker.

    python word_evidence.py --data-dir <train dir> --splits <splits.pkl> --shards <score shard dir>
                            --exclude-seed 11 --exclude-n 150000 --out word_evidence.pkl
"""
import argparse, glob, math, os, pickle, random, sys, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MIN_COUNT = 30


def main():
    import pandas as pd
    from norm import raw_name_tokens
    from features import token_diff
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--exclude-seed", type=int, default=11)
    ap.add_argument("--exclude-n", type=int, default=150000)
    ap.add_argument("--n", type=int, default=400000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.perf_counter()
    sp = pickle.load(open(a.splits, "rb"))
    matcher_ids = set(random.Random(a.exclude_seed).sample(sp["fit"], a.exclude_n))
    pool = [i for i in sp["fit"] if i not in matcher_ids]
    ids = set(random.Random(23).sample(pool, min(a.n, len(pool))))
    rd = lambda f: pd.read_csv(os.path.join(a.data_dir, f), sep="\t", dtype=str, keep_default_na=False)
    gt = rd("train_ground_truth.tsv")
    gt = {s: set(m.split(",")) if m else set() for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    s1 = rd("train_source1.tsv").set_index("entity_id").business_name
    names = pd.concat([rd("train_source2.tsv"), rd("train_source3.tsv")]).set_index("entity_id").business_name
    add = {True: Counter(), False: Counter()}; rm = {True: Counter(), False: Counter()}; n = Counter()
    for f in glob.glob(os.path.join(a.shards, "score_*.tsv")):
        for line in open(f, encoding="utf-8"):
            sid, tot, kept = line.rstrip("\n").split("\t")
            if sid not in ids or not kept:
                continue
            qt = raw_name_tokens(s1[sid])
            for x in kept.split(","):
                t = x.rsplit(":", 1)[0]
                y = t in gt[sid]
                ad, rv = token_diff(qt, raw_name_tokens(names[t]))
                add[y].update(set(ad)); rm[y].update(set(rv)); n[y] += 1
    out = {}
    for kind, tab in (("add", add), ("rm", rm)):
        e = {}
        for w in set(tab[True]) | set(tab[False]):
            c1, c0 = tab[True][w], tab[False][w]
            if c1 + c0 >= MIN_COUNT:
                v = math.log((c0 + 1) / (n[False] + 1)) - math.log((c1 + 1) / (n[True] + 1))
                e[w] = max(-4.0, min(8.0, v))
        out[kind] = e
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    top = sorted(out["add"].items(), key=lambda x: -x[1])[:25]
    print(f"{len(ids):,} S1, pairs true {n[True]:,} false {n[False]:,}; tokens add {len(out['add']):,} rm {len(out['rm']):,} "
          f"in {time.perf_counter() - t0:.0f}s")
    print("strongest sibling markers (add):", [(w, round(v, 2)) for w, v in top])


if __name__ == "__main__":
    main()
