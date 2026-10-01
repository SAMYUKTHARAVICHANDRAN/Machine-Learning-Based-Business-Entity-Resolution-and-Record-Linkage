"""Metric-aware match selection.

Macro-F0.5 is scored per S1 entity, so the right cut depends on the entity's whole
candidate list, not on each pair alone. For one entity with candidate probabilities
p_1 >= p_2 >= ..., choosing the top k gives (plug-in expectation)

    TP_k = p_1 + ... + p_k            T = sum of all p + m
    E[F_k] = 1.25 TP_k / (1.25 TP_k + 0.25 (T - TP_k) + (k - TP_k))      k >= 1
    E[F_0] = prod(1 - p_i) * exp(-m)  (probability the entity is a true singleton)

where m is the expected number of true links the candidate search missed. Probabilities
are sharpened as p**gamma before use; m and gamma are tuned on the dev split.
"""
import math


def select_expected_f(cands, m=0.1, gamma=1.0, floor=0.02, total=None):
    """cands: [(target_id, prob)]. Returns the chosen set of target ids.

    m=0.1, gamma=1 are the settings behind every reported v2 dev/holdout number (on v1
    dev scores, m in {0, 0.2} and gamma in {1, 1.5, 2} were within 0.005 of each other).
    `total` (sum of ALL candidate probabilities) lets callers pass only the candidates
    above `floor` while keeping the expected number of true links exact."""
    c = sorted(((t, p ** gamma) for t, p in cands if p >= floor), key=lambda x: -x[1])
    if not c:
        return set()
    if total is None:
        total = sum(p ** gamma for _, p in cands)
    T = total + m
    best_k = 0
    log_p0 = sum(math.log1p(-min(p, 0.999999)) for _, p in c) - m
    best = math.exp(log_p0)
    tp = 0.0
    for k, (_, p) in enumerate(c, 1):
        tp += p
        f = 1.25 * tp / (1.25 * tp + 0.25 * (T - tp) + (k - tp))
        if f > best:
            best, best_k = f, k
    return {t for t, _ in c[:best_k]}


def resolve_ownership(pred, scores):
    """Each S2/S3 record may belong to at most one S1 (training labels: 0 of 7,638,365
    linked records have two owners). Keep only the highest-probability claim.

    pred: {s1: set(targets)}; scores: {s1: [(target, prob)]}. Returns new pred."""
    prob = {s: dict(c) for s, c in scores.items()}
    owner = {}
    for s, ts in pred.items():
        for t in ts:
            p = prob[s][t]
            if t not in owner or p > owner[t][1]:
                owner[t] = (s, p)
    return {s: {t for t in ts if owner[t][0] == s} for s, ts in pred.items()}
