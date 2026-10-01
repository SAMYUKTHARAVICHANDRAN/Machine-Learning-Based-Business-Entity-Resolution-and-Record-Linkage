"""Count S1 records per (country, normalized-name key) for one split; saved next to that split's index.
Label-free: uses only the split's own Source-1 file (train S1 for training/dev, test S1 for test).

    python s1_counts.py --s1 <split>_source1.tsv --index <index dir>"""
import argparse, os, pickle, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__":
    import pandas as pd
    from norm import norm_name, base_norm
    from shared_index import name_key
    ap = argparse.ArgumentParser(); ap.add_argument("--s1", required=True); ap.add_argument("--index", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.s1, sep="\t", dtype=str, keep_default_na=False)
    c = Counter(cn + "|" + name_key(norm_name(n) or base_norm(n).strip()) for cn, n in zip(df.country, df.business_name))
    with open(os.path.join(a.index, "s1_namecount.pkl"), "wb") as f:
        pickle.dump(dict(c), f)
    print(f"{len(df):,} S1 -> {len(c):,} name keys; max {max(c.values())}")
