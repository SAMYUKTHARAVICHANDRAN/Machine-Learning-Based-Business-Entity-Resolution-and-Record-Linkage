# %% [1] Config — edit the two paths if needed. Needs TRIDENT_pipeline.ipynb to have run once (uses its cached parts).
import os, glob, subprocess, sys, time
from pathlib import Path
import numpy as np, duckdb, pyarrow as pa

ROOT      = Path(r"E:\Datasets\Amazon_ML_Challenge")
A_FILE    = Path(r"C:\Users\krith\Downloads\matching_results.tsv")        # your 0.973 submission
MY_FILE   = ROOT / "submission" / "output" / "matching_results.tsv"      # TRIDENT output (fixes the part-file row order)
PARTS     = sorted(glob.glob(str(ROOT / "work" / "trident_v2" / "infer_v2_*")))[-1]
TEST_S1   = ROOT / "student_resource" / "dataset" / "test" / "test_source1.tsv"
VALIDATOR = ROOT / "student_resource" / "utils" / "validate_submission.py"
OUT       = ROOT / "submission" / "variants"
OUT.mkdir(parents=True, exist_ok=True)
for p in (A_FILE, MY_FILE, TEST_S1, VALIDATOR, Path(PARTS)):
    assert Path(p).exists(), f"missing: {p}"
con = duckdb.connect()
con.execute(f"SET threads={max(1, (os.cpu_count() or 2) - 1)}")
t0 = time.time()

# %% [2] Load: test order + country, TRIDENT pair probabilities, A's links
def first_col(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        return [line.split("\t", 1)[0].strip() for line in f if line.strip()]

with open(TEST_S1, encoding="utf-8") as f:
    next(f)
    rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
TEST_IDS = [r[0].strip() for r in rows]                       # exact input order
COUNTRY = {r[0].strip(): (r[3].strip() if len(r) > 3 else "") for r in rows}
MY_ORDER = first_col(MY_FILE)                                 # row k of MY_FILE == qid k in the part files
assert sorted(MY_ORDER) == sorted(TEST_IDS), "TRIDENT output does not cover the test set"
con.register("s", pa.table({"qid": np.arange(len(MY_ORDER), dtype=np.int32), "s1": MY_ORDER,
                            "country": [COUNTRY[x] for x in MY_ORDER]}))
con.register("ord", pa.table({"pos": np.arange(len(TEST_IDS), dtype=np.int64), "s1": TEST_IDS}))
parts = PARTS.replace("\\", "/")
con.execute(f"""CREATE TABLE pr AS
  SELECT s.s1, p.cand_id AS t, p.p, s.country,
         row_number() OVER (PARTITION BY p.cand_id ORDER BY p.p DESC, p.qid) = 1 AS mybest
  FROM read_parquet('{parts}/*.pairs.parquet') p JOIN s USING (qid)""")
a_path = str(A_FILE).replace("\\", "/")
con.execute(f"""CREATE TABLE al AS SELECT DISTINCT trim(s1) AS s1, trim(t) AS t FROM (
  SELECT s1, unnest(string_split(m, ',')) AS t FROM read_csv('{a_path}', delim='\t', header=true, quote='', escape='',
         all_varchar=true, columns={{'s1':'VARCHAR','m':'VARCHAR'}}) WHERE coalesce(m, '') <> '') WHERE trim(t) <> ''""")
print(con.execute("SELECT (SELECT count(*) FROM pr), (SELECT count(*) FROM al)").fetchone(), f"{time.time()-t0:.0f}s")

# %% [3] One table with every link either source proposes + A's one-owner resolution (by TRIDENT probability)
con.execute("""CREATE TABLE u AS
  SELECT coalesce(pr.s1, al.s1) AS s1, coalesce(pr.t, al.t) AS t, pr.p, (al.t IS NOT NULL) AS ina,
         coalesce(pr.mybest, false) AS mybest
  FROM pr FULL JOIN al ON pr.s1 = al.s1 AND pr.t = al.t""")
con.execute("""CREATE TABLE ua AS
  SELECT u.*, row_number() OVER (PARTITION BY t ORDER BY coalesce(p, -1) DESC, s1) = 1 AS a_owner
  FROM u WHERE ina""")
print("A targets owned by >1 entity:", con.execute(
    "SELECT count(*) FROM (SELECT t FROM al GROUP BY t HAVING count(*) > 1)").fetchone()[0])

# %% [4] Build variants (each is a set of (s1, t) links)
con.execute("SET preserve_insertion_order=true")
V = {
  "V1_A_dedup":      "SELECT s1, t FROM ua WHERE a_owner",
  "V2_A_dedup_prune":"SELECT s1, t FROM ua WHERE a_owner AND NOT (p IS NOT NULL AND p < 0.02)",
  "V4_A_and_mine":   "SELECT s1, t FROM ua WHERE a_owner AND p >= 0.5",
}
con.execute(f"CREATE TABLE v2 AS {V['V2_A_dedup_prune']}")
V["V3_V2_plus_sure"] = """SELECT s1, t FROM v2 UNION ALL
  SELECT s1, t FROM u WHERE NOT ina AND p >= 0.99 AND mybest AND t NOT IN (SELECT t FROM v2)"""
con.execute(f"CREATE TABLE v3 AS {V['V3_V2_plus_sure']}")
V["V5_FR_A_else_V3"] = """SELECT v3.s1, v3.t FROM v3 JOIN s USING (s1) WHERE s.country <> 'France' UNION ALL
  SELECT ua.s1, ua.t FROM ua JOIN s USING (s1) WHERE ua.a_owner AND s.country = 'France'"""
V["V6_mine_t095"] = "SELECT s1, t FROM pr WHERE p >= 0.95 AND mybest"

def write_variant(name, sql):
    con.execute(f"CREATE OR REPLACE TABLE cur AS SELECT DISTINCT s1, t FROM ({sql})")
    dup = con.execute("SELECT count(*) FROM (SELECT t FROM cur GROUP BY t HAVING count(*) > 1)").fetchone()[0]
    m = dict(con.execute("SELECT s1, string_agg(t, ',' ORDER BY t) FROM cur GROUP BY s1").fetchall())
    extra = set(m) - set(TEST_IDS)
    assert not extra, f"{name}: unknown S1 ids {list(extra)[:3]}"
    path = OUT / f"{name}.tsv"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for e in TEST_IDS:
            f.write(f"{e}\t{m.get(e, '')}\n")
    n = con.execute("SELECT count(*) FROM cur").fetchone()[0]
    shared = con.execute("SELECT count(*) FROM cur JOIN al USING (s1, t)").fetchone()[0]
    print(f"{name:18s} links {n:>9,}  shared-with-A {shared:>9,}  new-vs-A {n - shared:>8,}  "
          f"multi-owner targets {dup}  empty rows {len(TEST_IDS) - len(m):,}")
    return path

PATHS = {k: write_variant(k, q) for k, q in V.items()}

# %% [5] Validate every file with the official validator (must print PASS)
for k, p in PATHS.items():
    r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(p), "--candidate", str(OUT / "none.tsv"),
                        "--test-dir", str(TEST_S1.parent)], capture_output=True, text=True)
    ok = r.returncode == 0 and "PASS" in r.stdout
    print(f"{k:18s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
print(f"all variants written to {OUT} in {time.time()-t0:.0f}s")
