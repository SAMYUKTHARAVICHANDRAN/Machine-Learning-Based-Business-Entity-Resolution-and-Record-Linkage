"""Simple merge rules that keep X's calibrated probabilities. Rules chosen on fold 0, reported on fold 1."""
import sys, pickle, random, hashlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(r"E:\Datasets\Amazon_ML_Challenge")
sys.path.insert(0, r"C:\Users\krith\Downloads\v3d_full_code\v3d_full_code\code\business_entity_resolution\src")
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
import trident_core as tc
from decide import select_expected_f
W = ROOT / "work" / "merge"
XS, TS = pickle.load(open(W / "x_scores.pkl", "rb")), pickle.load(open(W / "t_scores.pkl", "rb"))
gtdf = pd.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
EV = sorted(set(XS) | set(TS))
gtm = dict(zip(gtdf.source1_entity_id, gtdf.matched_entity_ids))
gt = {e: set(gtm[e].split(",")) if gtm[e] else set() for e in EV}
fold = {e: tc.digest_id("mfold:" + e) % 2 for e in EV}

def f05(P, T):
    tp = len(P & T); fp = len(P) - tp; fn = len(T) - tp
    return (1.0 if not P else 0.0) if not T else (5 * tp / (5 * tp + 4 * fp + fn) if tp else 0.0)

def run(rule, k):
    s = []
    for e in EV:
        if fold[e] != k:
            continue
        c = rule(dict(XS.get(e, [])), dict(TS.get(e, [])))
        s.append(f05(select_expected_f(list(c.items()), total=sum(c.values())), gt[e]))
    return np.array(s)

def x_only(px, pt): return px
def rescue(a, w):      # X probs kept; TRIDENT-only candidates added at w * pT when pT >= a
    return lambda px, pt: {**px, **{t: w * p for t, p in pt.items() if t not in px and p >= a}}
def veto(b):           # X probs kept, but damped when TRIDENT saw the pair and is confident it is wrong
    return lambda px, pt: {t: (p * 0.5 if (t in pt and pt[t] < b) else p) for t, p in px.items()}

rules = {"X": x_only}
for a in (0.9, 0.97, 0.99):
    for w in (0.5, 0.8, 1.0):
        rules[f"rescue a={a} w={w}"] = rescue(a, w)
for b in (0.005, 0.02):
    rules[f"veto b={b}"] = veto(b)
base0 = run(x_only, 0).mean()
res0 = {k: run(r, 0).mean() for k, r in rules.items()}
best = max(res0, key=res0.get)
for k, v in sorted(res0.items(), key=lambda kv: -kv[1])[:5]:
    print(f"fold0 {k:24s} {v:.5f} ({v - base0:+.5f})")
d = run(rules[best], 1) - run(x_only, 1)
rng = np.random.default_rng(2026)
b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
print(f"REPORT fold1: best-on-fold0 rule '{best}' gain vs X {d.mean():+.5f}  95% CI [{np.quantile(b,.025):+.5f}, {np.quantile(b,.975):+.5f}]")
