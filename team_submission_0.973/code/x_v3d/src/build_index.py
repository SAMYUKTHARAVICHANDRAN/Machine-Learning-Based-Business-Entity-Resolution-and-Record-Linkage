"""Build the shared, memory-mapped index over S2+S3 of one split.

    python build_index.py --data-dir <dir> --prefix train|test --out <index dir>
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_tuples(path):
    import pandas as pd
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return list(zip(df["entity_id"], df["business_name"], df["business_address"], df["country"]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--prefix", required=True, choices=["train", "test"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import norm
    from shared_index import build
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    log(f"aliases loaded: name={len(norm.NAME_ALIAS):,} addr={len(norm.ADDR_ALIAS):,}")
    t = time.perf_counter()
    recs = (load_tuples(os.path.join(a.data_dir, f"{a.prefix}_source2.tsv"))
            + load_tuples(os.path.join(a.data_dir, f"{a.prefix}_source3.tsv")))
    log(f"loaded {len(recs):,} records")
    build(recs, a.out, log=log)
    log(f"index built in {time.perf_counter() - t:.0f}s")
