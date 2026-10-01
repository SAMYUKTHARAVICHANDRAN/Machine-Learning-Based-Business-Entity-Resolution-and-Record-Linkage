"""Learn token alias tables from training links (fit split only).

Two mining rules, both over (S1 record, linked S2/S3 record) pairs:

1. Cross-script alignment. When exactly one side of a name pair is in an Indic script
   and both have the same token count, tokens align by position:
   "फ्यूचर कंस्ट्रक्शन प्राइवेट लिमिटेड" <-> "future construction private limited".
2. Difference co-occurrence. For pairs whose token sets differ by only a few tokens,
   the tokens present on one side only are counted against the tokens present on the
   other side only: "mh" <-> "maharashtra", "rd" <-> "road".

A variant v maps to canonical c when c is v's most frequent partner, the evidence is
at least MIN_COUNT, and c explains at least MIN_RATIO of v's appearances. Numbers are
never aliased. Random typos do not recur, so they do not reach the thresholds.

    python learn_aliases.py --data-dir <train dir> --splits <splits.pkl> --out aliases.pkl
"""
import argparse
import os
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INDIC = re.compile(r"[ऀ-෿]")
HAS_DIGIT = re.compile(r"\d")
MIN_COUNT = {"pos": 5, "diff": 20}
MIN_RATIO = 0.5
MIN_LEN = {"name": 3, "addr": 2}   # shorter Latin variants were truncation noise ("a" -> india)


def _subseq(short, long):
    it = iter(long)
    return all(ch in it for ch in short)


def _plausible(v, c, kind):
    """Same-script alias must look like an abbreviation or a typo of its canonical form.
    Without this, component omissions teach city -> state ("bhayandar" -> maharashtra),
    which erases locality evidence."""
    from rapidfuzz import fuzz
    if kind == "addr" and bool(INDIC.search(v)) != bool(INDIC.search(c)):
        return True                     # v3: Indic-script state/district names -> Latin form
    if len(v) < MIN_LEN[kind]:
        return False
    short, long = (v, c) if len(v) <= len(c) else (c, v)
    return (short[0] == long[0] and _subseq(short, long)) or fuzz.ratio(v, c) >= 70


def _select(pair_counts, token_counts, min_count, kind=None):
    best = {}
    for (v, c), n in pair_counts.items():
        if n > best.get(v, (None, 0.0))[1]:
            best[v] = (c, n)
    table = {}
    for v, (c, n) in best.items():
        if (n >= min_count and n / token_counts[v] >= MIN_RATIO and v != c
                and not HAS_DIGIT.search(v) and not HAS_DIGIT.search(c)
                and (kind is None or _plausible(v, c, kind))):
            table[v] = (c, n)
    # break 2-cycles (a->b and b->a): keep the better-supported direction
    for v in list(table):
        if v in table:
            c, n = table[v]
            if c in table and table[c][0] == v:
                if table[c][1] > n:
                    del table[v]
                else:
                    del table[c]
    return {v: c for v, (c, _) in table.items()}


def mine(pairs):
    """pairs: iterable of (s1_name, s1_addr, t_name, t_addr). Returns (name, addr, stats)."""
    import norm
    norm.NAME_ALIAS.clear()
    norm.ADDR_ALIAS.clear()
    pos, pos_n = Counter(), Counter()
    name_co, name_n = defaultdict(float), Counter()
    addr_co, addr_n = defaultdict(float), Counter()

    def diff_count(A, B, co, cnt, max_diff):
        dA, dB = set(A) - set(B), set(B) - set(A)
        if 1 <= len(dA) <= max_diff and 1 <= len(dB) <= max_diff:
            w = 1.0 / len(dA)
            for b in dB:
                cnt[b] += 1
                for a in dA:
                    co[(b, a)] += w

    for s1n, s1a, tn, ta in pairs:
        A, B = norm.raw_name_tokens(s1n), norm.raw_name_tokens(tn)
        ia, ib = bool(INDIC.search(s1n)), bool(INDIC.search(tn))
        if ia != ib:
            if len(A) == len(B):
                lat, ind = (A, B) if ib else (B, A)
                for l, i in zip(lat, ind):
                    pos[(i, l)] += 1
                    pos_n[i] += 1
        elif A != B:
            diff_count(A, B, name_co, name_n, 2)
        diff_count(norm.raw_addr_tokens(s1a), norm.raw_addr_tokens(ta), addr_co, addr_n, 3)

    translit = _select(pos, pos_n, MIN_COUNT["pos"])
    name = _select(name_co, name_n, MIN_COUNT["diff"], "name")
    name.update(translit)                       # alignment wins for Indic tokens
    addr = _select(addr_co, addr_n, MIN_COUNT["diff"], "addr")
    stats = {"translit": len(translit), "name": len(name), "addr": len(addr)}
    return name, addr, stats


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--links", type=int, default=2_000_000)
    args = ap.parse_args()
    t0 = time.perf_counter()
    rd = lambda f: pd.read_csv(os.path.join(args.data_dir, f), sep="\t", dtype=str,
                               keep_default_na=False)

    fit = set(pickle.load(open(args.splits, "rb"))["fit"])
    gt = rd("train_ground_truth.tsv")
    gt = gt[gt.source1_entity_id.isin(fit) & (gt.matched_entity_ids != "")]
    links = gt.assign(t=gt.matched_entity_ids.str.split(",")).explode("t")[["source1_entity_id", "t"]]
    links = links.sample(n=min(args.links, len(links)), random_state=0)
    s1 = rd("train_source1.tsv").set_index("entity_id")
    need = set(links.t)
    tg = pd.concat([ch[ch.entity_id.isin(need)]
                    for f in ("train_source2.tsv", "train_source3.tsv")
                    for ch in pd.read_csv(os.path.join(args.data_dir, f), sep="\t", dtype=str,
                                          keep_default_na=False, chunksize=1_000_000)]).set_index("entity_id")
    a = s1.loc[links.source1_entity_id]
    b = tg.loc[links.t]
    pairs = zip(a.business_name.values, a.business_address.values,
                b.business_name.values, b.business_address.values)
    print(f"loaded {len(links):,} fit links in {time.perf_counter() - t0:.0f}s", flush=True)

    t1 = time.perf_counter()
    name, addr, stats = mine(pairs)
    print(f"mined in {time.perf_counter() - t1:.0f}s: {stats}", flush=True)
    with open(args.out, "wb") as f:
        pickle.dump({"name": name, "addr": addr, "stats": stats}, f)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
