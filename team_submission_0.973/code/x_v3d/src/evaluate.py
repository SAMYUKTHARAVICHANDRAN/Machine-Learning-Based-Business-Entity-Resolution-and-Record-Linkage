"""Official macro-F0.5 (per S1 entity, singletons and zero-candidate entities included)."""
import numpy as np


def f05(pred, true):
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    return 1.25 * tp / (1.25 * tp + 0.25 * (len(true) - tp) + (len(pred) - tp))


def evaluate(pred_sets, ids, gt):
    """pred_sets: {s1: set(targets)}. Returns (macro F0.5, pair precision, pair recall)."""
    tot = TP = FP = FN = 0
    for i in ids:
        pred, true = pred_sets.get(i, set()), gt[i]
        tot += f05(pred, true)
        tp = len(pred & true)
        TP, FP, FN = TP + tp, FP + len(pred) - tp, FN + len(true) - tp
    return tot / len(ids), TP / max(TP + FP, 1), TP / max(TP + FN, 1)


def threshold_sets(scores, thr):
    return {s: {t for t, p in c if p >= thr} for s, c in scores.items()}


def report(scores, ids, gt, thrs=(.3, .4, .5, .6, .7, .8), log=print):
    cand = {s: {t for t, _ in c} for s, c in scores.items()}
    links = sum(len(gt[i]) for i in ids)
    rec = sum(len(cand.get(i, set()) & gt[i]) for i in ids) / links
    oracle = np.mean([f05(cand.get(i, set()) & gt[i], gt[i]) for i in ids])
    nc = [len(cand.get(i, ())) for i in ids]
    log(f"  candidate recall {rec:.2%}  oracle F0.5 {oracle:.4f}  "
        f"cands/S1 mean {np.mean(nc):.0f} p95 {np.percentile(nc, 95):.0f} "
        f"p99 {np.percentile(nc, 99):.0f} max {max(nc)}")
    best = (None, -1)
    for t in thrs:
        f, P, R = evaluate(threshold_sets(scores, t), ids, gt)
        log(f"  thr {t:.2f}  F0.5 {f:.4f}  P {P:.3f}  R {R:.3f}")
        if f > best[1]:
            best = (t, f)
    log(f"  BEST thr {best[0]:.2f}  F0.5 {best[1]:.4f}")
    return best
