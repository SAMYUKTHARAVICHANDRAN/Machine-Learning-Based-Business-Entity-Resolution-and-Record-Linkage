"""Train the two-stage matcher (v3b) on the fit split, then score and evaluate dev.

Disjoint fit-split S1 sets (no S1 contributes labels to two components):
  A  stage-1 pair model          random.Random(11).sample(fit, N_A)
  B  word-evidence table         random.Random(23).sample(fit - A, 400k)   (word_evidence.py)
  C  stage-2 context model       random.Random(31).sample(fit - A - B, N_C), scored out of sample by
                                 the stage-1 model trained on A

    python train2.py --data-dir <train dir> --splits <splits.pkl> --index <index dir> --work <dir>
"""
import argparse, glob, os, pickle, random, sys, time
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


def stack(shards, key, gt):
    Xs, ys, ss, ts = [], [], [], []
    for p in shards:
        z = np.load(p)
        if len(z["t"]):
            Xs.append(z[key]); ss.append(z["s1"]); ts.append(z["t"])
            ys.append(np.fromiter((t in gt[s] for s, t in zip(z["s1"], z["t"])), bool, len(z["t"])))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss), np.concatenate(ts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n-a", type=int, default=150_000)
    ap.add_argument("--n-c", type=int, default=150_000)
    ap.add_argument("--tag", default="v3b")
    a = ap.parse_args()

    import pairs
    from features import FEAT_NAMES
    from stage2 import STAGE2_NAMES
    from matcher import PairMatcher
    from sklearn.ensemble import HistGradientBoostingClassifier

    os.makedirs(a.work, exist_ok=True)
    s1 = {r[0]: r for r in load_tuples(os.path.join(a.data_dir, "train_source1.tsv"))}
    gt = load_gt(os.path.join(a.data_dir, "train_ground_truth.tsv"))
    sp = pickle.load(open(a.splits, "rb"))
    A = random.Random(11).sample(sp["fit"], a.n_a)
    Aset = set(A)
    poolB = [i for i in sp["fit"] if i not in Aset]
    B = set(random.Random(23).sample(poolB, 400_000))
    poolC = [i for i in poolB if i not in B]
    C = random.Random(31).sample(poolC, a.n_c)
    assert not (Aset & B) and not (Aset & set(C)) and not (B & set(C))
    log(f"S1 sets: A {len(A):,}  B {len(B):,}  C {len(C):,} (disjoint)")

    model_path = os.path.join(a.work, f"model_{a.tag}.pkl")
    s1_path = os.path.join(a.work, f"model_{a.tag}_stage1.pkl")
    if not os.path.isfile(s1_path):
        shard_dir = os.path.join(a.work, "pairs_A")
        shards = sorted(glob.glob(os.path.join(shard_dir, "pairs_*.npz"))) or \
            pairs.generate([s1[i] for i in A], a.index, shard_dir, log=log)
        X, y, _, _ = stack(shards, "X", gt)
        log(f"stage-1 matrix {X.shape}, positives {y.sum():,} ({y.mean():.2%})")
        t = time.perf_counter()
        m1 = PairMatcher().fit(X, y)
        log(f"stage 1 trained in {time.perf_counter() - t:.0f}s ({m1.clf.n_iter_} iterations)")
        with open(s1_path, "wb") as f:
            pickle.dump({"model": m1, "features": FEAT_NAMES}, f)
        del X, y
    ctx_dir = os.path.join(a.work, "ctx_C")
    shards = sorted(glob.glob(os.path.join(ctx_dir, "ctx_*.npz"))) or \
        pairs.generate([s1[i] for i in C], a.index, ctx_dir, log=log, model_path=s1_path)
    Cx, cy, _, _ = stack(shards, "C", gt)
    log(f"stage-2 matrix {Cx.shape}, positives {cy.sum():,} ({cy.mean():.2%})")
    t = time.perf_counter()
    m2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                        l2_regularization=1.0, random_state=0).fit(Cx, cy)
    log(f"stage 2 trained in {time.perf_counter() - t:.0f}s ({m2.n_iter_} iterations)")
    m1 = pickle.load(open(s1_path, "rb"))["model"]
    with open(model_path, "wb") as f:
        pickle.dump({"model": m1, "stage2": m2, "features": FEAT_NAMES, "stage2_features": STAGE2_NAMES}, f)
    for name, path in (("stage1", s1_path), ("two-stage", model_path)):
        sc = pairs.score([s1[i] for i in sp["dev"]], a.index, path, log=log)
        with open(os.path.join(a.work, f"scores_{a.tag}_{name}_dev.pkl"), "wb") as f:
            pickle.dump(sc, f)
    log("done")
