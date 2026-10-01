"""Learn address-component aliases (v3d): whole comma-separated components that the generator swaps.

For a pair of linked records, components are compared after normalization; when exactly one S1
component and exactly one target component have no fuzzy counterpart, the pair (target comp -> S1 comp)
is counted. A target component is aliased to its most frequent partner when seen >= MIN_COUNT times
and that partner explains >= MIN_SHARE of its unmatched occurrences. This learns multi-word state
names <-> codes ("north carolina" <-> "nc"), place <-> parent municipality (hyannis -> barnstable) and,
from unlabelled confident matches, France department <-> region (gironde -> nouvelle aquitaine).

  labelled   : python learn_components.py labelled --data-dir <train> --splits <splits.pkl> --out comp_alias.pkl
  unlabelled : python learn_components.py unlabelled --data-dir <test> --shards <score shards> --country France
                                           --table comp_alias.pkl --out comp_alias_test.pkl
"""
import argparse, glob, os, pickle, random, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MIN_COUNT, MIN_SHARE = 30, 0.5
DIG = re.compile(r"\d")
STREET_START = {"r", "rue", "av", "ave", "avenue", "bd", "blvd", "boulevard", "all", "allee", "imp", "impasse", "rte",
                "route", "pl", "place", "ch", "chemin", "q", "quai", "crs", "cours", "sq", "square", "res",
                "residence", "road", "rd", "street", "st", "lane", "ln", "drive", "dr", "marg", "gali"}


def comps(addr):
    import norm
    s = norm.undot(norm.strip_po(norm.base_norm(addr)))
    return [k for k in (norm.comp_key(c) for c in s.split(",")) if k]


def unmatched(q, t):
    from rapidfuzz.distance import Levenshtein as Lev
    def close(a, others):
        return any(a == o or (len(a) > 3 and Lev.distance(a, o) <= max(1, len(a) // 5)) or
                   (len(a) > 4 and (a in o or o in a)) for o in others)
    Q = [c for c in q if not DIG.search(c) and len(c) >= 2]
    T = [c for c in t if not DIG.search(c) and len(c) >= 2]
    return [c for c in Q if not close(c, T)], [c for c in T if not close(c, Q)]


def select(pairs, occ):
    best = {}
    for (tu, qu), n in pairs.items():
        if n > best.get(tu, (None, 0))[1]:
            best[tu] = (qu, n)
    table = {tu: qu for tu, (qu, n) in best.items() if n >= MIN_COUNT and n / occ[tu] >= MIN_SHARE and tu != qu
             and tu.split()[0] not in STREET_START and qu.split()[0] not in STREET_START}   # admin areas, not streets
    for tu in list(table):                   # break 2-cycles, keep the better-supported direction
        qu = table.get(tu)
        if qu in table and table[qu] == tu:
            if pairs[(tu, qu)] >= pairs[(qu, tu)]:
                del table[qu]
            else:
                del table[tu]
    return table


def mine(link_iter):
    pairs, occ = Counter(), Counter()
    for qa, ta in link_iter:
        qu, tu = unmatched(comps(qa), comps(ta))
        for c in tu:
            occ[c] += 1
        if len(qu) == 1 and len(tu) == 1:
            pairs[(tu[0], qu[0])] += 1
    return pairs, occ


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["labelled", "unlabelled"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits")
    ap.add_argument("--shards")
    ap.add_argument("--country", default="France")
    ap.add_argument("--table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300_000)
    a = ap.parse_args()
    rd = lambda f: pd.read_csv(os.path.join(a.data_dir, f), sep="\t", dtype=str, keep_default_na=False)
    if a.mode == "labelled":
        sp = pickle.load(open(a.splits, "rb"))
        ids = set(random.Random(41).sample(sp["fit"], a.n))
        gt = rd("train_ground_truth.tsv"); gt = gt[gt.source1_entity_id.isin(ids) & (gt.matched_entity_ids != "")]
        s1 = rd("train_source1.tsv").set_index("entity_id")
        tg = pd.concat([rd("train_source2.tsv"), rd("train_source3.tsv")]).set_index("entity_id").business_address
        table, pairs = {}, Counter()
        for c in sorted(set(s1.country)):
            sub = gt[gt.source1_entity_id.map(s1.country) == c]
            it = ((s1.at[s, "business_address"], tg[t]) for s, m in zip(sub.source1_entity_id, sub.matched_entity_ids)
                  for t in m.split(","))
            pc, occ = mine(it)
            table[c] = select(pc, occ); pairs.update(pc)
            print(f"{c}: {len(table[c])} component aliases")
    else:
        s1 = rd("test_source1.tsv"); s1 = s1[s1.country == a.country].set_index("entity_id").business_address
        tg = pd.concat([rd("test_source2.tsv"), rd("test_source3.tsv")]).set_index("entity_id").business_address
        def it():
            for f in glob.glob(os.path.join(a.shards, "score_*.tsv")):
                for line in open(f, encoding="utf-8"):
                    sid, tot, kept = line.rstrip("\n").split("\t")
                    if sid in s1.index and kept:
                        for x in kept.split(","):
                            t, p = x.rsplit(":", 1)
                            if float(p) >= 0.99:
                                yield s1[sid], tg[t]
        pairs, occ = mine(it())
        table = pickle.load(open(a.table, "rb")) if a.table else {}
        new = select(pairs, occ)
        for k in ("pas de calais", "pas-de-calais"):
            cand = [(q, n) for (t, q), n in pairs.items() if t == k]
            print(f"  {k!r}: unmatched {occ[k]}, top partners {sorted(cand, key=lambda x: -x[1])[:3]}")
        table[a.country] = {**table.get(a.country, {}), **new}
        print(f"{a.country} (label-free) added {len(new)}: {sorted(new.items())[:40]}")
    with open(a.out, "wb") as f:
        pickle.dump(table, f)
    print(f"component aliases per country: { {c: len(t) for c, t in table.items()} } -> {a.out}")
    for c, t in table.items():
        print(c, sorted(t.items(), key=lambda kv: -pairs[(kv[0], kv[1])])[:12])


if __name__ == "__main__":
    main()
