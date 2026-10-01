"""Train the pair matcher on fit-split pairs, then score and evaluate the dev split.

    python train.py --data-dir <train dir> --splits <splits.pkl> --index <index dir>
                    --work <dir> [--n-fit 100000] [--tag v2]
"""
import argparse, os, pickle, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_tuples(path):
    import pandas as pd
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return list(zip(df["entity_id"], df["business_name"], df["business_address"], df["country"]))


def load_gt(path):
    import pandas as pd
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {s: set(m.split(",")) if m else set()
            for s, m in zip(df["source1_entity_id"], df["matched_entity_ids"])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n-fit", type=int, default=100_000)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--eval-split", default="dev")
    a = ap.parse_args()

    import pairs
    from features import FEAT_NAMES
    from matcher import PairMatcher
    from evaluate import report

    os.makedirs(a.work, exist_ok=True)
    s1 = {r[0]: r for r in load_tuples(os.path.join(a.data_dir, "train_source1.tsv"))}
    gt = load_gt(os.path.join(a.data_dir, "train_ground_truth.tsv"))
    sp = pickle.load(open(a.splits, "rb"))
    fit_ids = random.Random(11).sample(sp["fit"], a.n_fit)

    shard_dir = os.path.join(a.work, f"pairs_{a.tag}")
    shards = pairs.generate([s1[i] for i in fit_ids], a.index, shard_dir, log=log)
    Xs, ys = [], []
    for p in shards:
        z = np.load(p)
        if len(z["t"]):
            Xs.append(z["X"])
            ys.append(np.fromiter((t in gt[s] for s, t in zip(z["s1"], z["t"])), bool, len(z["t"])))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    log(f"train matrix {X.shape}, positives {y.sum():,} ({y.mean():.2%}), "
        f"{X.nbytes / 1e9:.2f} GB")

    t = time.perf_counter()
    m = PairMatcher().fit(X, y)
    log(f"trained in {time.perf_counter() - t:.0f}s ({m.clf.n_iter_} iterations)")
    model_path = os.path.join(a.work, f"model_{a.tag}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": m, "threshold": 0.6, "features": FEAT_NAMES}, f)
    del X, y, Xs, ys

    ids = sp[a.eval_split]
    scores = pairs.score([s1[i] for i in ids], a.index, model_path, log=log)
    with open(os.path.join(a.work, f"scores_{a.tag}_{a.eval_split}.pkl"), "wb") as f:
        pickle.dump(scores, f)
    log(f"{a.eval_split} evaluation ({len(ids):,} S1):")
    thr, f05 = report(scores, ids, gt, log=log)
    with open(model_path, "wb") as f:
        pickle.dump({"model": m, "threshold": thr, "features": FEAT_NAMES}, f)
    log(f"saved {model_path} with threshold {thr}")
