"""Label-free word evidence for countries without training labels (France in this test set).

The labelled table (word_evidence.py) cannot know French sibling markers ("groupe", "holding",
"participations" ...). On the unlabelled test catalogue of such a country we contrast, among a previous
model's scored candidates,
  proxy-true:    p >= 0.99 and the same main house number as the S1 record
  proxy-sibling: name token_set >= 80 but a different main house number
and add tokens that are added to the S1 name >= MIN_RATIO times more often in proxy-siblings, seen at
least MIN_COUNT times, with their proxy log-odds. Tokens already in the labelled table are left alone,
so features of labelled countries are unchanged.

    python word_evidence_fr.py --test-dir <test dir> --shards <test score shards> --country France
                               --table word_evidence.pkl --out word_evidence_test.pkl
"""
import argparse, glob, math, os, pickle, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MIN_COUNT, MIN_RATIO = 200, 8.0


def main():
    import pandas as pd
    from rapidfuzz import fuzz
    from norm import raw_name_tokens
    from features import token_diff, main_number
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--country", default="France")
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rd = lambda f: pd.read_csv(os.path.join(a.test_dir, f), sep="\t", dtype=str, keep_default_na=False)
    s1 = rd("test_source1.tsv"); s1 = s1[s1.country == a.country].set_index("entity_id")
    tg = pd.concat([rd("test_source2.tsv"), rd("test_source3.tsv")])
    tg = tg[tg.country == a.country].set_index("entity_id")
    pos, neg = Counter(), Counter(); npos = nneg = 0
    for f in glob.glob(os.path.join(a.shards, "score_*.tsv")):
        for line in open(f, encoding="utf-8"):
            sid, tot, kept = line.rstrip("\n").split("\t")
            if sid not in s1.index or not kept:
                continue
            qn, qa = s1.at[sid, "business_name"], s1.at[sid, "business_address"]
            qt, qm = raw_name_tokens(qn), main_number(qa)
            if qm is None:
                continue
            for x in kept.split(","):
                t, p = x.rsplit(":", 1); p = float(p)
                tn, ta = tg.at[t, "business_name"], tg.at[t, "business_address"]
                tm = main_number(ta)
                if tm is None:
                    continue
                add, _ = token_diff(qt, raw_name_tokens(tn))
                if p >= 0.99 and tm == qm:
                    pos.update(set(add)); npos += 1
                elif tm != qm and fuzz.token_set_ratio(qn.casefold(), tn.casefold()) >= 80:
                    neg.update(set(add)); nneg += 1
    tab = pickle.load(open(a.table, "rb"))
    added = {}
    for w, c0 in neg.items():
        if w in tab["add"] or c0 < MIN_COUNT:
            continue
        r0, r1 = c0 / nneg, (pos[w] + 1) / npos
        if r0 / r1 >= MIN_RATIO:
            added[w] = max(-4.0, min(8.0, math.log(r0 / r1)))
    tab["add"].update(added)
    tab.setdefault("unlabelled", {})[a.country] = added
    with open(a.out, "wb") as f:
        pickle.dump(tab, f)
    print(f"{a.country}: proxy-true {npos:,} proxy-sibling {nneg:,}; added {len(added)} tokens: "
          f"{sorted(added.items(), key=lambda x: -x[1])}")


if __name__ == "__main__":
    main()
