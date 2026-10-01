"""Stage 2: re-score each candidate using the rest of its S1's candidate list (v3b).

The audit (work/audit/DIAGNOSIS.md) found the pair model is calibrated but ambiguous: ~21k dev pairs sit
at p in [0.1, 0.9) and only 42% of them are true. The generator gives every entity one corrupted
address per source (all S2 copies of an entity share "1107", all S3 copies share "1108."), while its
negative siblings carry their own number. Whether a candidate agrees with the entity's confident
records is therefore evidence the pair model cannot see. For every candidate with stage-1 p >= FLOOR:

  p1, rank            stage-1 probability and its rank in the list
  hi_same, hi_other   confident (p>=0.9) other candidates from the same / the other source
  sup_same/_other     max p of other candidates (same / other source) sharing this main house number
  con_same/_other     max p of other candidates (same / other source) with a DIFFERENT main number
  name_sup            max p of other candidates whose name is near-identical (token_set >= 90)
  sib_crowd           near-identical names with a different main number (sibling crowd size)
  addr_dup            max p of other candidates with the identical normalized address
  mnum_eq_s1, t_empty main number equals the S1's; candidate address empty
  sum_p, n_05, max_other   list-level context

A gradient-boosted model on these (trained on fit-split S1 disjoint from the stage-1 and the
word-evidence S1) outputs the final probability used by decide.select_expected_f.
"""
import numpy as np
from rapidfuzz import fuzz

from features import main_number

FLOOR = 0.02
STAGE2_NAMES = ["p1", "rank", "hi_same", "hi_other", "sup_same", "con_same", "sup_other", "con_other",
                "name_sup", "sib_crowd", "addr_dup", "mnum_eq_s1", "t_empty", "sum_p", "n_05", "max_other",
                "n_cands_floor",
                # v3c: S1-side name counts (label-free; from the split's own S1 file) and stage-1 features
                "q_s1cnt", "c_s1cnt", "c_s1cnt_other",
                "n_tokset", "a_tokset", "ev_add_max", "mnum_rel", "t_addr_empty", "q_addr_empty",
                "n_exact_norm", "a_num_contra"]
S1_FEATS = [0, 9, 38, 45, 35, 34, 16, 33]     # indices into features.FEAT_NAMES


def context_rows(s1_raw_addr, cands, extra=None):
    """cands: list of (tid, p, raw_addr, norm_name, norm_addr) with p >= FLOOR. Returns feature rows.
    extra: optional per-candidate list of additional values appended to each row (v3c)."""
    k = len(cands)
    if not k:
        return np.zeros((0, len(STAGE2_NAMES)), np.float32)
    q_m = main_number(s1_raw_addr)
    P = np.array([c[1] for c in cands], dtype=np.float64)
    src = [c[0][:2] for c in cands]
    mn = [main_number(c[2]) for c in cands]
    nn = [c[3] for c in cands]
    na = [c[4] for c in cands]
    order = np.argsort(-P, kind="stable")
    rank = np.empty(k, dtype=np.int64); rank[order] = np.arange(k)
    sim = np.zeros((k, k), dtype=bool)
    for i in range(k):
        for j in range(i + 1, k):
            if fuzz.token_set_ratio(nn[i], nn[j]) >= 90:
                sim[i, j] = sim[j, i] = True
    sum_p, n05 = float(P.sum()), float((P >= 0.5).sum())
    rows = []
    for i in range(k):
        hs = ho = 0
        ss = cs = so = co = ns = ad = mo = 0.0
        crowd = 0
        for j in range(k):
            if j == i:
                continue
            pj, same = P[j], src[j] == src[i]
            mo = max(mo, pj)
            if pj >= 0.9:
                if same:
                    hs += 1
                else:
                    ho += 1
            if mn[i] is not None and mn[j] is not None:
                if mn[j] == mn[i]:
                    if same:
                        ss = max(ss, pj)
                    else:
                        so = max(so, pj)
                else:
                    if same:
                        cs = max(cs, pj)
                    else:
                        co = max(co, pj)
            if sim[i, j]:
                ns = max(ns, pj)
                if mn[i] is not None and mn[j] is not None and mn[j] != mn[i]:
                    crowd += 1
            if na[i] and na[i] == na[j]:
                ad = max(ad, pj)
        rows.append([P[i], float(rank[i]), float(hs), float(ho), ss, cs, so, co, ns, float(crowd), ad,
                     1.0 if (q_m is not None and mn[i] == q_m) else 0.0, 1.0 if not na[i] else 0.0,
                     sum_p, n05, mo, float(k)] + (list(extra[i]) if extra is not None else []))
    return np.asarray(rows, dtype=np.float32)
