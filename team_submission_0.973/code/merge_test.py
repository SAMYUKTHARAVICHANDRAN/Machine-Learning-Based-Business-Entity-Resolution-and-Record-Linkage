"""Final merged submission = X v3d + TRIDENT rescue (US/India only).
Rescue rule (measured +0.00095 on held-out train, 95% CI +0.0006..+0.0014): keep every X probability;
add candidates X never retrieved when TRIDENT p >= 0.9; then X's expected-F0.5 selection + one-owner rule."""
import sys, glob, os, hashlib, subprocess
from pathlib import Path
import duckdb, pyarrow as pa, numpy as np
ROOT = Path(r"E:\Datasets\Amazon_ML_Challenge")
XSRC = Path(r"C:\Users\krith\Downloads\v3d_full_code\v3d_full_code\code\business_entity_resolution\src")
sys.path.insert(0, str(XSRC))
from decide import select_expected_f, resolve_ownership
XT = ROOT / "work" / "merge" / "x_test"
TEST = ROOT / "student_resource" / "dataset" / "test"
OUT = ROOT / "submission" / "merged"; OUT.mkdir(parents=True, exist_ok=True)
A, W, RESCUE_COUNTRIES = 0.9, 1.0, {"US", "India"}

md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
print("X v3d output md5 here:", md5(XT / "matching_results.tsv")[:8], "(X's documented v3d: 2e104188)")

# test order + country
with open(TEST / "test_source1.tsv", encoding="utf-8") as f:
    next(f); rows = [l.rstrip("\n").split("\t") for l in f if l.strip()]
IDS = [r[0] for r in rows]; CTRY = {r[0]: r[3] for r in rows}

# X: all candidates, kept probabilities (>= 0.02) and total per S1
xc, xs, xt = {}, {}, {}
for p in sorted(glob.glob(str(XT / "shards" / "cand_*.tsv"))):
    for l in open(p, encoding="utf-8"):
        s, ids = l.rstrip("\n").split("\t"); xc[s] = ids.split(",") if ids else []
for p in sorted(glob.glob(str(XT / "shards" / "score_*.tsv"))):
    for l in open(p, encoding="utf-8"):
        s, tot, kept = l.rstrip("\n").split("\t")
        xs[s] = [(x.rsplit(":", 1)[0], float(x.rsplit(":", 1)[1])) for x in kept.split(",")] if kept else []
        xt[s] = float(tot)
assert set(xc) == set(IDS) == set(xs), "X shards do not cover the test set"

# TRIDENT: high-confidence pairs (qid order = rows of TRIDENT's matching_results.tsv)
my_order = [l.split("\t", 1)[0] for l in open(ROOT / "submission/output/matching_results.tsv", encoding="utf-8").read().splitlines()[1:]]
parts = sorted(glob.glob(str(ROOT / "work/trident_v2/infer_v2_*")))[-1].replace("\\", "/")
con = duckdb.connect()
con.register("o", pa.table({"qid": np.arange(len(my_order), dtype=np.int32), "s1": my_order}))
hi = con.execute(f"SELECT o.s1, p.cand_id, p.p FROM read_parquet('{parts}/*.pairs.parquet') p JOIN o USING (qid) WHERE p.p >= {A}").fetchall()

added = {}
for s, t, p in hi:
    if CTRY[s] in RESCUE_COUNTRIES and t not in set(xc[s]):
        added.setdefault(s, []).append((t, W * float(p)))
print(f"rescued candidates: {sum(map(len, added.values())):,} for {len(added):,} S1")

scores, pred = {}, {}
for s in IDS:
    c = xs[s] + added.get(s, [])
    scores[s] = c
    pred[s] = select_expected_f(c, total=xt[s] + sum(p for _, p in added.get(s, [])))
n0 = sum(map(len, pred.values())); pred = resolve_ownership(pred, scores); n1 = sum(map(len, pred.values()))
print(f"selected {n0:,} links; one-owner removed {n0 - n1:,}")

with open(OUT / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s in IDS:
        f.write(f"{s}\t{','.join(sorted(pred[s]))}\n")
with open(OUT / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s in IDS:
        ids = xc[s] + [t for t, _ in added.get(s, [])]
        assert pred[s] <= set(ids)
        f.write(f"{s}\t{','.join(ids)}\n")
r = subprocess.run([sys.executable, str(ROOT / "student_resource/utils/validate_submission.py"), "--matching", str(OUT / "matching_results.tsv"),
                    "--candidate", str(OUT / "candidate_pairs.tsv"), "--test-dir", str(TEST)], capture_output=True, text=True)
print(r.stdout[-600:])
if r.returncode != 0:
    raise SystemExit("VALIDATION FAILED")
