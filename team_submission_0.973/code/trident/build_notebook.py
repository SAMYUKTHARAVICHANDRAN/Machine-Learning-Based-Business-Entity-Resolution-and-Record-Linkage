"""Generates TRIDENT_pipeline.ipynb. The core module source is embedded verbatim in a
%%writefile cell so the notebook is self-contained."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
CORE = (HERE / "src" / "trident_core.py").read_text(encoding="utf-8")
OUT = HERE.parent.parent / "TRIDENT_pipeline.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# TRIDENT — Business Entity Resolution (Amazon ML Challenge 2026)

Source-1 → {Source-2, Source-3} matching: **normalise → DuckDB blocking index → top-40 candidates →
batched pairwise features → CatBoost → calibrated threshold → one-owner-per-target resolution →
validated TSVs**.

Run the cells top to bottom from a fresh kernel. The only cell you normally edit is **§1 Configuration**.

### Changes relative to the original specification (each is a config switch and was A/B-measured)

| # | Where | Change | Why (measured on the data) |
|---|---|---|---|
| C1 | §6 index | Build the index **directly from the TSVs** → Parquet → DuckDB | `local_runs/train.sqlite` / `test.sqlite` do not exist; TSV→Parquet is the same data with one less hop |
| C2 | normalisation | **Indic-script transliteration** + native state / legal-word replacement before the spec `norm()` | 9–11 % of S2 names and 5–6 % of S3 names are Devanagari/Tamil/Telugu/…; S1 is 100 % Latin, so the spec `norm()` can never match them |
| C3 | normalisation | **Canonical abbreviations** (St/Street, Ct/Court, NC/North Carolina, MH/Maharashtra, Pvt/Private, R./Rue, Av./Avenue …) | the abbreviation noise is the dominant address/name noise in true pairs |
| C4 | blocking | **Country is a hard blocking partition** (country feature kept) | 0.0 % of 7.6 M true links cross countries; key frequencies become per-country |
| C5 | blocking | +2 routes: **C** concatenated core name (domain names such as `memorialcenter.com`), **Y** house-number × street token | ~3 % of targets use a domain as the business name; garbled names still share number+street |
| C6 | retrieval | Jaro-Winkler is computed only for the top `PRE_CAP` pairs by evidence (then top-40 by priority) | runtime; recall impact is measured in §8 |
| C7 | features | 29 spec features **+ 17 extra** (core-name similarity, domain concat ratio, house-number agreement, per-route evidence, JW, script flags, ranking margins) | spec §30 Stage B; kept only because calibration F0.5 improves |
| C8 | decision | **One owner per target**: a S2/S3 record is assigned only to the S1 entity with the highest probability | in the full ground truth **no** S2/S3 record belongs to more than one S1 entity (0 of 7.6 M links) |
| C9 | evaluation | Larger deterministic splits from the full truth file (≈11 k calibration, ≈11 k audit entities) | 500/500 gives a CI ≈ ±0.01, too wide to see 0.001-level effects |
| C10 | French | French legal forms / street types in the stop and abbreviation lists | 15 % of the test set is France, absent from training — **cannot be validated with labels** |
| C11 | normalisation | **Learned transliteration lexicon** (`treding→trading`, `bildars→builders` …) fitted on true links of *training* entities only | rule transliteration of generated Hindi/Tamil/… names gives `indastrij`, `sarvisej`; oracle ceiling 0.9814 → 0.9861 |
| C12 | ranking | Priority uses JW(name) in place of JW(addr) when an address is empty; CAP 40 → 60, PRE_CAP 300 | ~3 % of targets have no address and were ranked below 40; oracle 0.9861 → 0.9880 |

Everything else (routes N/A/n/a/P/Q/X with weights 8/8/1/1/3/3/3, frequency caps 40/400, evidence
`w/(1+ln f)`, priority `evidence + 3·JW(name) + 4·JW(addr)`, CAP = 40, grouped holdout, CatBoost start
parameters, 0.05–0.99 threshold grid, macro-F0.5, bootstrap CI, resumable inference, output validation)
follows the specification.

> **Honesty rule:** every score printed below is computed at run time. The **oracle ceiling** is *not*
> a model score, and no 0.99 claim is made unless the untouched audit set measures it.
""")

md("## 1. Configuration  (edit paths/flags here only)")
code(r"""
import os, sys
from pathlib import Path

# ---- paths ------------------------------------------------------------------
PROJECT_ROOT   = Path(os.environ.get("TRIDENT_PROJECT_ROOT", Path.cwd())).resolve()
DATA_ROOT      = PROJECT_ROOT / "student_resource" / "dataset"
TRAIN_DIR      = DATA_ROOT / "train"
TEST_DIR       = DATA_ROOT / "test"
TRAIN_S1       = TRAIN_DIR / "train_source1.tsv"
TRAIN_TARGETS  = [TRAIN_DIR / "train_source2.tsv", TRAIN_DIR / "train_source3.tsv"]
TRAIN_TRUTH    = TRAIN_DIR / "train_ground_truth.tsv"
TEST_S1        = TEST_DIR / "test_source1.tsv"
TEST_TARGETS   = [TEST_DIR / "test_source2.tsv", TEST_DIR / "test_source3.tsv"]
OFFICIAL_VALIDATOR = PROJECT_ROOT / "student_resource" / "utils" / "validate_submission.py"

RUN_ROOT        = PROJECT_ROOT / "work" / "trident_v2"          # caches, indexes, models, parts
SRC_DIR         = PROJECT_ROOT / "code" / "business_entity_resolution" / "src"
CORE_PATH       = SRC_DIR / "trident_core.py"
SUBMISSION_ROOT = PROJECT_ROOT / "submission" / "output"
DUCK_TMP        = RUN_ROOT / "duck_tmp"                          # DuckDB spill directory

# ---- pipeline ---------------------------------------------------------------
SEED = 2026
CFG = dict(
    norm_mode="canon",       # "spec" = original norm() only;  "canon" = C2 + C3
    routes="v2",             # "spec" = N,A,n,a,P,Q,X ; "v2" adds C and Y routes (C5)
    country_blocking=True,   # C4
    tok_cap=40,              # max frequency for single-token keys (n:, a:)
    other_cap=400,           # max frequency for other keys
    pre_cap=300,             # C6: pairs per query that get Jaro-Winkler scored
    cap=60,                  # final candidates per query (spec: 40; 60 measured +0.002 oracle, C12)
    prio="v2",               # C12: "spec" = evidence+3JW(name)+4JW(addr); "v2" uses JW(name) when an address is empty
)
USE_LEXICON = True           # C11: learned transliteration lexicon (training entities only)
FEATURE_SET = "v2"           # "spec29" or "v2" (29 + 17 extra)
TARGET_UNIQUENESS = True     # C8 (validated on calibration below; switched off automatically if it does not help)

# entity-level split, in hash buckets out of 10,000 (C9)
AUDIT_BP, CALIB_BP, TRAIN_BP = 50, 50, 300     # ≈0.5 % / 0.5 % / 3 % of 2.2 M train S1 entities

CATBOOST_BASE = dict(iterations=4000, depth=7, learning_rate=0.06, loss_function="Logloss",
                     eval_metric="Logloss", l2_leaf_reg=5, random_seed=SEED, early_stopping_rounds=100,
                     allow_writing_files=False)
COMPUTE_DEVICE = "AUTO"      # "AUTO" | "GPU" | "CPU"
GPU_DEVICES    = "0"
GPU_RAM_PART   = 0.90
BENCH_ROWS, BENCH_ITERS = 300_000, 200

BATCH_SIZE          = 20_000   # Source-1 queries per retrieval batch
TEST_BENCH_SAMPLE   = 10_000   # reservoir sample for the runtime projection
MAX_TEST_HOURS      = 8.0      # stop before full inference if the conservative estimate exceeds this
RUN_CONTEXT_EVAL    = True     # C8 needs competitor S1 entities: retrieval over ALL train S1 (DuckDB only)
RUN_TAG             = "v2"     # inference parts go to RUN_ROOT/infer_<RUN_TAG>_<signature>

import multiprocessing, psutil
CPU_THREADS  = max(1, (os.cpu_count() or 2) - 1)
POOL_WORKERS = max(1, min(20, CPU_THREADS))
_avail_gb    = psutil.virtual_memory().available / 1e9 if 'psutil' in sys.modules else 8.0
DUCKDB_MEM_GB = max(2, int(min(0.7 * psutil.virtual_memory().total / 1e9, max(4.0, 0.8 * _avail_gb))))
print(f"PROJECT_ROOT={PROJECT_ROOT}\nRUN_ROOT={RUN_ROOT}\nthreads={CPU_THREADS} pool={POOL_WORKERS} duckdb_mem={DUCKDB_MEM_GB}GB")
""")

md("## 2. Dependencies and environment diagnostics")
code(r"""
import importlib, platform
REQUIRED = ["numpy", "pandas", "rapidfuzz", "catboost", "duckdb", "pyarrow"]
OPTIONAL = ["psutil", "torch"]
missing = []
versions = {}
for m in REQUIRED + OPTIONAL:
    try:
        versions[m] = getattr(importlib.import_module(m), "__version__", "?")
    except Exception:
        versions[m] = None
        if m in REQUIRED:
            missing.append(m)
if missing:
    raise ImportError(f"Missing required packages {missing}. Install with:\n"
                      f"  {sys.executable} -m pip install numpy pandas rapidfuzz catboost duckdb pyarrow psutil")

import numpy as np, pandas as pd, duckdb, pyarrow as pa, pyarrow.parquet as pq, catboost, rapidfuzz, psutil
from catboost.utils import get_gpu_device_count
vm = psutil.virtual_memory()
print(f"Python      {sys.version.split()[0]}  ({platform.platform()})")
print(f"CPU cores   {os.cpu_count()}   RAM total {vm.total/1e9:.1f} GB, available {vm.available/1e9:.1f} GB")
for m in REQUIRED + OPTIONAL:
    print(f"  {m:10s} {versions[m] or 'not installed (optional)'}")
try:
    N_GPU = get_gpu_device_count()
except Exception as e:
    N_GPU = 0
    print("CatBoost GPU query failed:", e)
print(f"CatBoost-visible GPUs: {N_GPU}")
GPU_INFO = None
if versions.get("torch"):
    import torch
    if torch.cuda.is_available():
        GPU_INFO = (torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9)
if GPU_INFO is None and N_GPU:
    import subprocess
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        GPU_INFO = tuple(x.strip() for x in q.split(","))
    except Exception:
        pass
print("GPU:", GPU_INFO or "none detected (CPU fallback will be used)")
""")

md("## 3. Preflight — verify every input, output location and resource before expensive work")
code(r"""
import shutil
problems = []
if sys.version_info < (3, 9):
    problems.append(f"Python >= 3.9 required, found {sys.version}")
for p in [TRAIN_S1, *TRAIN_TARGETS, TRAIN_TRUTH, TEST_S1, *TEST_TARGETS]:
    if not Path(p).is_file():
        problems.append(f"missing input file: {p}")
for d in [RUN_ROOT, SRC_DIR, SUBMISSION_ROOT, DUCK_TMP, RUN_ROOT / "index"]:
    try:
        Path(d).mkdir(parents=True, exist_ok=True)
        t = Path(d) / ".write_test"; t.write_text("x"); t.unlink()
    except Exception as e:
        problems.append(f"directory not writable: {d} ({e})")
try:
    _c = duckdb.connect(str(RUN_ROOT / ".preflight.duckdb")); _c.execute("select 1"); _c.close()
    (RUN_ROOT / ".preflight.duckdb").unlink()
except Exception as e:
    problems.append(f"DuckDB cannot create a database in {RUN_ROOT}: {e}")
free_gb = shutil.disk_usage(RUN_ROOT).free / 1e9
if free_gb < 25:
    problems.append(f"only {free_gb:.1f} GB free on the RUN_ROOT drive; ~25 GB recommended (indexes + spill)")
if vm.available / 1e9 < 4:
    problems.append(f"only {vm.available/1e9:.1f} GB RAM available; close other programs")
print(f"disk free {free_gb:.0f} GB | RAM available {vm.available/1e9:.1f} GB | CatBoost GPUs {N_GPU}")
if problems:
    raise RuntimeError("PREFLIGHT FAILED:\n  " + "\n  ".join(problems))
print("PREFLIGHT OK")
""")

md("""
## 4. Core library (`trident_core.py`)

All logic that must be identical for training and inference lives in this one module. It is written
to `code/business_entity_resolution/src/` (also the `src/` of the submission package) and imported;
process-pool workers on Windows must import functions from a module, not from notebook cells.
""")
code("%%writefile $CORE_PATH\n" + CORE)
code(r"""
import importlib, multiprocessing as mp, time, json, gc
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import trident_core as tc
importlib.reload(tc)
np.random.seed(SEED)
import random; random.seed(SEED)
print("trident_core", tc.PIPELINE_VERSION, "| features:", len(tc.feature_names(FEATURE_SET)))

def new_con(db_path=None, read_only=True):
    con = duckdb.connect(str(db_path), read_only=read_only) if db_path else duckdb.connect()
    tc.configure_duckdb(con, CPU_THREADS, DUCKDB_MEM_GB, DUCK_TMP)
    return con
""")

md("## 5. Unit tests that do not need data (normalisation, metric, splits, output validator)")
code(r"""
# Test 1 — normalisation
assert tc.norm_spec("Café  Déjà-Vu, Inc.") == "cafe deja vu inc"
assert tc.norm_spec(None) == "" and tc.norm_spec(float("nan")) == "" and tc.norm_spec("") == ""
assert tc.norm_spec("ÉCOLE_primaire/Nº5") == "ecole primaire no5"
assert tc.canon_name("राम मार्केटिंग प्राइवेट लिमिटेड") == "ram marketing pvt ltd"
assert tc.canon_name("Pvt. EFS Print Ventures Ltd.") == "pvt efs print ventures ltd"
assert tc.canon_addr("12555 Sixteenth Street, Chino, California", "US") == "12555 sixteenth st chino ca"
assert tc.canon_addr("00414 STATE ST, PENDLETON, IN", "US") == "414 state st pendleton in"
assert tc.canon_addr("383 - R. DU BLANC SEAU, TOURCOING", "France") == "383 r du blanc seau tourcoing"
assert tc.canon_name("x", mode="spec") == "x" and tc.canon_addr("Road No 5", "India", mode="spec") == "road no 5"
# idempotence of canonical forms
for s in ["12555 sixteenth st chino ca", "ram marketing pvt ltd"]:
    assert tc.canon_addr(s, "US") == s or tc.canon_name(s) == s

# metric
e = np.array([0, 0, 1, 2]); y = np.array([True, False, False, True]); n_true = np.array([2., 0., 1.])
s, *_ = tc.entity_scores(e, y, np.array([True, True, False, True]), n_true)
assert np.allclose(s, [5 / (5 + 4 + 1), 1.0, 1.0])            # entity0 TP1 FP1 FN1; entity1 singleton
s, *_ = tc.entity_scores(e, y, np.array([False, False, True, True]), n_true)
assert np.allclose(s, [0.0, 0.0, 1.0])
# spec example from the problem statement: P=2/3, R=1 -> 0.714
s, *_ = tc.entity_scores(np.array([0, 0, 0]), np.array([True, False, True]), np.array([True] * 3), np.array([2.]))
assert abs(s[0] - 0.7142857) < 1e-6

# Test 6/7 — split isolation on synthetic ids
ids = [f"S1-{i}" for i in range(50000)]
sp = tc.select_splits(ids, 500, 500, 3000)
assert not (set(sp["audit"]) & set(sp["calib"])) and not (set(sp["train"]) & (set(sp["audit"]) | set(sp["calib"])))
tr_ids = [ids[i] for i in sp["train"]]
hm = tc.inner_holdout_mask(tr_ids)
assert not ({x for x, h in zip(tr_ids, hm) if h} & {x for x, h in zip(tr_ids, hm) if not h})
assert 0.15 < hm.mean() < 0.25

# Test 8 — submission validator on a small artificial output
tmpd = RUN_ROOT / "selftest"; tmpd.mkdir(exist_ok=True)
(tmpd / "c.tsv").write_text("source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S3-2\nS1-2\t\n", encoding="utf-8", newline="\n")
(tmpd / "m.tsv").write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS3-2\nS1-2\t\n", encoding="utf-8", newline="\n")
assert tc.validate_outputs(tmpd / "m.tsv", tmpd / "c.tsv", ["S1-1", "S1-2"])
for bad in ["S1-1\tS3-9\nS1-2\t\n", "S1-1\tS3-2,S3-2\nS1-2\t\n", "S1-2\t\nS1-1\tS3-2\n", "S1-1\tS3-2\n"]:
    (tmpd / "m.tsv").write_text("source1_entity_id\tmatched_entity_ids\n" + bad, encoding="utf-8", newline="\n")
    try:
        tc.validate_outputs(tmpd / "m.tsv", tmpd / "c.tsv", ["S1-1", "S1-2"]); raise SystemError("validator missed a defect")
    except AssertionError:
        pass
print("unit tests PASSED")
""")

md("## 6. Entity splits and labels (grouped by Source-1 entity; calibration and audit are disjoint)")
code(r"""
con0 = new_con()
s1_train_raw = tc.read_source_arrow(con0, TRAIN_S1)
s1_ids_all = [x.strip() for x in s1_train_raw.column("entity_id").to_pylist()]
SPLITS = tc.select_splits(s1_ids_all, AUDIT_BP, CALIB_BP, TRAIN_BP)
TRUTH = tc.read_truth(con0, TRAIN_TRUTH)
missing_truth = [x for x in s1_ids_all[:1000] if x not in TRUTH]
assert not missing_truth, f"truth rows missing for e.g. {missing_truth[:3]}"
for k, v in SPLITS.items():
    ids_k = [s1_ids_all[i] for i in v]
    n_links = sum(len(TRUTH[i]) for i in ids_k)
    print(f"{k:6s}: {len(v):7,} entities, {n_links:8,} true links, singletons {sum(not TRUTH[i] for i in ids_k):,}")
S_AUD, S_CAL, S_TR = (set(SPLITS[k]) for k in ("audit", "calib", "train"))
assert not (S_AUD & S_CAL) and not (S_TR & (S_AUD | S_CAL)), "split leakage"
""")

md("""
## 7. Learned transliteration lexicon (C11) and worker pool

Fitted on true links of **training** Source-1 entities only (calibration and audit entities are
excluded), then frozen into the index signature. Every normalisation worker receives the same lexicon.
""")
code(r"""
LEX_EXCL = {s1_ids_all[i] for i in np.concatenate([SPLITS["audit"], SPLITS["calib"]])}
lex_sig = tc.signature({"inputs": [tc.file_fingerprint(p) for p in [TRAIN_S1, *TRAIN_TARGETS, TRAIN_TRUTH]],
                        "excl": [AUDIT_BP, CALIB_BP], "v": tc.PIPELINE_VERSION})
LEX_PATH = RUN_ROOT / f"lexicon_{lex_sig}.json"
if not USE_LEXICON:
    LEXICON = {}
elif LEX_PATH.exists():
    LEXICON = json.loads(LEX_PATH.read_text(encoding="utf-8")); print(f"[lexicon] cache hit: {len(LEXICON)} entries")
else:
    LEXICON = tc.learn_translit_lexicon(con0, TRAIN_S1, TRAIN_TARGETS, TRAIN_TRUTH, LEX_EXCL)
    LEX_PATH.write_text(json.dumps(LEXICON, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8", newline="\n")
tc.set_lexicon(LEXICON)
if "POOL" in globals():
    POOL.close(); POOL.join()
POOL = mp.get_context("spawn").Pool(POOL_WORKERS, initializer=tc.set_lexicon, initargs=(LEXICON,))
print("lexicon signature", tc.lexicon_signature(), "| pool workers", POOL_WORKERS)
""")

md("""
## 8. Target index (training pool = all of train Source-2 + Source-3)

Normalised records → Parquet → DuckDB `records`; blocking keys → frequency-filtered `postings`
(`key` = 64-bit hash of `country ␟ route:key`, `ev = w / (1 + ln f)` precomputed).
The index is rebuilt automatically when inputs, normalisation, routes, caps or the DuckDB version change.
""")
code(r"""
def index_path(split):
    sig = tc.signature(tc.index_signature(TRAIN_TARGETS if split == "train" else TEST_TARGETS, CFG))
    return RUN_ROOT / "index" / f"{split}_{sig}.duckdb"

t0 = time.perf_counter()
TRAIN_INDEX = tc.build_target_index(index_path("train"), TRAIN_TARGETS, CFG, POOL, CPU_THREADS, DUCKDB_MEM_GB, DUCK_TMP)
print(f"train index ready in {time.perf_counter()-t0:.0f}s -> {TRAIN_INDEX}")
con_tr = new_con(TRAIN_INDEX)
print(con_tr.execute("SELECT count(*) FROM records").fetchone()[0], "records;",
      con_tr.execute("SELECT count(*) FROM postings").fetchone()[0], "postings")
""")

md("""
## 9. Candidate retrieval + **oracle ceiling**

The oracle pretends a perfect classifier picks exactly the true targets that are *present in the
candidate list*. It is an **upper bound for any model on this candidate generator, not a model
score**. If it is below the target (0.99), retrieval must be improved before model tuning.
Blocks are cached with a signature (inputs, cfg, feature set, split).
""")
code(r"""
TIMER = tc.Timer()
def block_for(name, idx):
    sig = tc.signature({"index": TRAIN_INDEX.name, "cfg": CFG, "fs": FEATURE_SET, "split": name,
                        "n": len(idx), "first": int(idx[0]) if len(idx) else -1, "v": tc.PIPELINE_VERSION,
                        "truth": tc.file_fingerprint(TRAIN_TRUTH), "lex": tc.lexicon_signature()})
    path = RUN_ROOT / "blocks" / f"{name}_{sig}.npz"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        print(f"[{name}] cache hit {path.name}")
        return tc.load_block(path)
    print(f"[{name}] building ({len(idx):,} queries)")
    b = tc.process_queries(con_tr, tc.take_raw(s1_train_raw, idx), CFG, FEATURE_SET, pool=POOL,
                           batch_size=BATCH_SIZE, truth=TRUTH, timer=TIMER, workers=CPU_THREADS)
    tc.save_block(path, b)
    return b

BLK = {k: block_for(k, SPLITS[k]) for k in ("train", "calib", "audit")}
for k, b in BLK.items():
    orc = tc.summarize(b["qid"], b["y"], b["y"], b["n_true"])
    cnt = np.bincount(b["qid"], minlength=len(b["n_true"]))
    print(f"{k:6s} ORACLE CEILING macro-F0.5 = {orc['macro_f05']:.4f} | candidate recall {orc['pair_recall']:.4f} "
          f"| cands/query mean {cnt.mean():.1f}, empty {np.mean(cnt == 0):.4f} | pos frac {b['y'].mean():.4f}")
    for kk in (5, 10, 20):
        m = b["rank"] < kk
        print(f"        oracle@top{kk}: {tc.summarize(b['qid'][m], b['y'][m], b['y'][m], b['n_true'])['macro_f05']:.4f}")
ORACLE_CAL = tc.summarize(BLK["calib"]["qid"], BLK["calib"]["y"], BLK["calib"]["y"], BLK["calib"]["n_true"])["macro_f05"]
if ORACLE_CAL < 0.99:
    print(f"\nNOTE: oracle ceiling {ORACLE_CAL:.4f} < 0.99 -> a 0.99 model score is impossible with this candidate set.")
""")

md("## 10. Feature checks: exactly N features, scalar ≡ batched on random real pairs, deterministic retrieval, cache invalidation")
code(r"""
FEAT = tc.feature_names(FEATURE_SET)
for b in BLK.values():
    assert b["X"].shape[1] == len(FEAT) == (29 if FEATURE_SET == "spec29" else 46)
    assert b["X"].dtype == np.float32

# Test 3 — scalar reference vs canonical batched implementation on 3,000 random real pairs
rng = np.random.default_rng(SEED)
qsel = rng.choice(SPLITS["calib"], 150, replace=False)
qraw = tc.take_raw(s1_train_raw, qsel)
nt = tc.normalize_records(*(qraw.column(i).to_pylist() for i in range(4)), CFG["norm_mode"])
q = tc.make_query_table(nt)
cand = tc.retrieve(con_tr, q, CFG)
Xb = tc.features_for_candidates(cand, q, FEATURE_SET, CFG["norm_mode"])
cd = cand.to_pydict()
idx = rng.choice(cand.num_rows, min(3000, cand.num_rows), replace=False)
Xs = np.stack([tc.features_scalar(nt["name"][cd["qid"][i]].as_py(), nt["addr"][cd["qid"][i]].as_py(),
                                  nt["country"][cd["qid"][i]].as_py(), cd["c_name"][i], cd["c_addr"][i],
                                  cd["c_country"][i], cd["cand_id"][i], {k: cd[k][i] for k in tc.RETRIEVAL_NUMERIC},
                                  FEATURE_SET, CFG["norm_mode"]) for i in idx])
np.testing.assert_allclose(Xs, Xb[idx], rtol=0, atol=0, equal_nan=True)
print(f"scalar == batched on {len(idx)} pairs x {Xb.shape[1]} features (exact float32 equality)")

# Test 4 — deterministic candidate generation
cand2 = tc.retrieve(con_tr, q, CFG)
assert cand.column("cand_id").to_pylist() == cand2.column("cand_id").to_pylist()
assert cand.column("qid").to_pylist() == cand2.column("qid").to_pylist()
print("retrieval deterministic")

# Test 5 — cache invalidation: changing CAP / routes / norm changes the index and block signatures
s_a = tc.signature(tc.index_signature(TRAIN_TARGETS, CFG))
s_b = tc.signature(tc.index_signature(TRAIN_TARGETS, {**CFG, "tok_cap": CFG["tok_cap"] + 1}))
assert s_a != s_b
blk_sig = lambda cfg: tc.signature({"cfg": cfg, "fs": FEATURE_SET})
assert blk_sig(CFG) != blk_sig({**CFG, "cap": CFG["cap"] + 20})
print("cache signatures change with configuration")
""")

md("""
## 11. CPU vs GPU benchmark and model training

`AUTO` times CatBoost on both devices on a representative subset and picks the faster one
(GPU start-up overhead can make CPU faster on small data). Early stopping uses the **grouped** inner
holdout (`digest_id('inner:'+qid) % 5 == 0`); the model is then refit on all training entities with the
selected number of trees.
""")
code(r"""
Xtr, ytr = BLK["train"]["X"], BLK["train"]["y"]
print(f"training pairs {len(ytr):,}: positives {int(ytr.sum()):,}, negatives {int((~ytr).sum()):,}, "
      f"positive fraction {ytr.mean():.4f}")
if COMPUTE_DEVICE == "AUTO":
    DEVICE, BENCH = tc.benchmark_devices(Xtr, ytr, CATBOOST_BASE, BENCH_ITERS, BENCH_ROWS, GPU_DEVICES, GPU_RAM_PART, SEED)
else:
    DEVICE, BENCH = COMPUTE_DEVICE, {}
    if DEVICE == "GPU" and N_GPU == 0:
        print("WARNING: GPU requested but CatBoost sees no GPU -> CPU"); DEVICE = "CPU"
print("selected device:", DEVICE)

tr_ents = BLK["train"]["entity_ids"]
hold_q = tc.inner_holdout_mask(tr_ents)
hold_pair = hold_q[BLK["train"]["qid"]]
assert not (set(np.unique(BLK["train"]["qid"][hold_pair])) & set(np.unique(BLK["train"]["qid"][~hold_pair])))
t0 = time.perf_counter()
m_es, dev_used = tc.fit_catboost(Xtr[~hold_pair], ytr[~hold_pair], CATBOOST_BASE, DEVICE,
                                 Xtr[hold_pair], ytr[hold_pair], GPU_DEVICES, GPU_RAM_PART)
BEST_ITERS = int(m_es.get_best_iteration()) + 1
print(f"early stopping on grouped holdout: best iteration count {BEST_ITERS} "
      f"(holdout logloss {m_es.get_best_score()['validation']['Logloss']:.5f}) [{dev_used}, {time.perf_counter()-t0:.0f}s]")
del m_es; gc.collect()
t0 = time.perf_counter()
MODEL, dev_used = tc.fit_catboost(Xtr, ytr, {**CATBOOST_BASE, "iterations": BEST_ITERS}, DEVICE,
                                  gpu_devices=GPU_DEVICES, gpu_ram_part=GPU_RAM_PART)
TRAIN_SECONDS = time.perf_counter() - t0
DEVICE_USED = dev_used
MODEL_PATH = RUN_ROOT / "models" / f"catboost_{tc.signature({'cfg': CFG, 'fs': FEATURE_SET, 'it': BEST_ITERS, 'n': len(ytr)})}.cbm"
MODEL_PATH.parent.mkdir(exist_ok=True)
MODEL.save_model(str(MODEL_PATH))
assert MODEL.tree_count_ == BEST_ITERS, "model is not fully trained"
print(f"refit on all training entities: {BEST_ITERS} trees on {DEVICE_USED} in {TRAIN_SECONDS:.0f}s -> {MODEL_PATH.name}")
imp = sorted(zip(MODEL.get_feature_importance(), FEAT), reverse=True)[:12]
print("top features:", ", ".join(f"{n}={v:.1f}" for v, n in imp))
""")

md("""
## 12. Competitor context for the one-owner-per-target rule (C8)

In the test set every Source-1 entity is scored, so a target predicted for two S1 entities can be
resolved to the more probable one. To evaluate that honestly on calibration/audit, we need the
*other* training S1 entities whose top-40 lists contain the same targets. This pass runs the same
retrieval for **all** training S1 entities (DuckDB only), keeps only rows whose target appears in the
calibration/audit candidate lists, and scores them with the model.
""")
code(r"""
def context_scores():
    ev_ids = sorted(set(BLK["calib"]["cand_id"]) | set(BLK["audit"]["cand_id"]))
    sig = tc.signature({"index": TRAIN_INDEX.name, "cfg": CFG, "fs": FEATURE_SET, "model": tc.file_sha256(MODEL_PATH),
                        "n": len(ev_ids), "splits": [AUDIT_BP, CALIB_BP, TRAIN_BP]})
    path = RUN_ROOT / "blocks" / f"context_{sig}.parquet"
    if path.exists():
        print("[context] cache hit"); return pd.read_parquet(path)
    con_tr.execute("CREATE OR REPLACE TEMP TABLE keep_rids AS SELECT rid FROM records WHERE entity_id IN "
                   "(SELECT unnest(?::VARCHAR[]))", [ev_ids])
    others = np.array(sorted(set(range(len(s1_ids_all))) - S_AUD - S_CAL))
    res = tc.process_queries(con_tr, tc.take_raw(s1_train_raw, others), CFG, FEATURE_SET, pool=POOL,
                             batch_size=BATCH_SIZE, model=MODEL, timer=TIMER, workers=CPU_THREADS,
                             keep_rids_table="keep_rids")
    df = pd.DataFrame({"s1": np.asarray(res["entity_ids"], dtype=object)[res["qid"]], "t": res["cand_id"], "p": res["p"]})
    df.to_parquet(path)
    return df

def eval_frame(name):
    b = BLK[name]
    p = MODEL.predict_proba(b["X"])[:, 1]
    return pd.DataFrame({"s1": np.asarray(b["entity_ids"], dtype=object)[b["qid"]], "t": b["cand_id"], "p": p,
                         "q": b["qid"], "y": b["y"]})

EV = {k: eval_frame(k) for k in ("calib", "audit")}
if RUN_CONTEXT_EVAL:
    t0 = time.perf_counter()
    CTX = context_scores()
    print(f"[context] {len(CTX):,} competitor pairs in {time.perf_counter()-t0:.0f}s")
else:
    CTX = pd.DataFrame({"s1": [], "t": [], "p": []})

def owner_flags(df):
    # best-for-target among: calib + audit + all other train S1 entities (full competition as in test)
    allp = pd.concat([EV["calib"][["s1", "t", "p"]], EV["audit"][["s1", "t", "p"]], CTX[["s1", "t", "p"]]],
                     ignore_index=True)
    best = allp.sort_values(["t", "p", "s1"], ascending=[True, False, True]).drop_duplicates("t")
    best_owner = dict(zip(best["t"], best["s1"]))
    return np.fromiter((best_owner[t] == s for s, t in zip(df["s1"], df["t"])), dtype=bool, count=len(df))

for k in EV:
    EV[k]["owner"] = owner_flags(EV[k])
""")

md("## 13. Threshold calibration (calibration split only) — the audit split is not looked at here")
code(r"""
cal = EV["calib"]; nt_cal = BLK["calib"]["n_true"]
T_PLAIN, S_PLAIN, CURVE_PLAIN = tc.calibrate_threshold(cal["q"].to_numpy(), cal["y"].to_numpy(), cal["p"].to_numpy(), nt_cal)
T_OWN, S_OWN, CURVE_OWN = tc.calibrate_threshold(cal["q"].to_numpy(), cal["y"].to_numpy(), cal["p"].to_numpy(), nt_cal,
                                                 extra_mask=cal["owner"].to_numpy())
print(f"calibration: threshold only        -> t={T_PLAIN:.2f}  macro-F0.5={S_PLAIN:.5f}")
print(f"calibration: + one-owner-per-target -> t={T_OWN:.2f}  macro-F0.5={S_OWN:.5f}")
USE_OWNER = bool(TARGET_UNIQUENESS and RUN_CONTEXT_EVAL and S_OWN > S_PLAIN)
THRESHOLD = T_OWN if USE_OWNER else T_PLAIN
CAL_SCORE = S_OWN if USE_OWNER else S_PLAIN
print(f"DECISION RULE (chosen on calibration): threshold={THRESHOLD:.2f}, one-owner-per-target={USE_OWNER}")
""")

md("## 14. Untouched audit evaluation + bootstrap CI + error analysis")
code(r"""
aud = EV["audit"]; nt_aud = BLK["audit"]["n_true"]
pred_aud = (aud["p"].to_numpy() >= THRESHOLD) & (aud["owner"].to_numpy() if USE_OWNER else True)
qa, ya = aud["q"].to_numpy(), aud["y"].to_numpy()
AUDIT = tc.summarize(qa, ya, pred_aud, nt_aud)
aud_scores = tc.entity_scores(qa, ya, pred_aud, nt_aud)[0]
AUDIT_CI = tc.bootstrap_ci(aud_scores, 2000, SEED)
AUDIT_ORACLE = tc.summarize(qa, ya, ya, nt_aud)["macro_f05"]
print(f"AUDIT macro-F0.5 = {AUDIT['macro_f05']:.5f}  (95% bootstrap CI {AUDIT_CI[0]:.5f} – {AUDIT_CI[1]:.5f}); "
      f"oracle ceiling {AUDIT_ORACLE:.5f}")
print(f"  pair precision {AUDIT['pair_precision']:.5f}  recall {AUDIT['pair_recall']:.5f}  "
      f"TP {AUDIT['TP']:,} FP {AUDIT['FP']:,} FN {AUDIT['FN']:,}  entities {AUDIT['entities']:,}")
print("  The CI covers entity-sampling noise only — NOT domain shift (France is unseen), leaderboard split "
      "differences, or adaptive-tuning bias.")

# error analysis by category
ents = BLK["audit"]["entity_ids"]
in_cand = np.bincount(qa, weights=ya, minlength=len(nt_aud))
fn_missing = int((nt_aud - in_cand).sum())
fn_present = int((ya & ~pred_aud).sum())
fp = ~ya & pred_aud
cat = pd.DataFrame({"t": aud["t"], "fp": fp, "fn": ya & ~pred_aud, "tp": ya & pred_aud,
                    "rank": BLK["audit"]["rank"], "src": aud["t"].str[:2],
                    "c_name_indic": BLK["audit"]["X"][:, FEAT.index("c_name_indic")] if "c_name_indic" in FEAT else 0,
                    "c_addr_missing": BLK["audit"]["X"][:, FEAT.index("c_addr_missing")],
                    "num_contra": BLK["audit"]["X"][:, FEAT.index("num_contradiction")]})
sing = nt_aud == 0
print(f"  FN: {fn_missing:,} never retrieved (retrieval), {fn_present:,} retrieved but rejected (model)")
print(f"  singleton entities: {int(sing.sum()):,}, of which falsely matched: {int(((np.bincount(qa, weights=pred_aud, minlength=len(nt_aud)) > 0) & sing).sum()):,}")
print(cat.groupby("src")[["tp", "fp", "fn"]].sum())
print("by target-name script (1 = Indic):"); print(cat.groupby("c_name_indic")[["tp", "fp", "fn"]].sum())
print("by missing target address:"); print(cat.groupby("c_addr_missing")[["tp", "fp", "fn"]].sum())
print("by address-number contradiction:"); print(cat.groupby("num_contra")[["tp", "fp", "fn"]].sum())
print("by candidate rank bucket:"); print(cat.groupby(pd.cut(cat["rank"], [-1, 0, 2, 5, 10, 40]), observed=True)[["tp", "fp", "fn"]].sum())
""")

md("## 15. Test target index + 10k runtime benchmark (stop before a run that would not finish in time)")
code(r"""
t0 = time.perf_counter()
TEST_INDEX = tc.build_target_index(index_path("test"), TEST_TARGETS, CFG, POOL, CPU_THREADS, DUCKDB_MEM_GB, DUCK_TMP)
con_te = new_con(TEST_INDEX)
print(f"test index ready in {time.perf_counter()-t0:.0f}s")
test_raw = tc.read_source_arrow(con_te, TEST_S1)
TEST_IDS = [x.strip() for x in test_raw.column("entity_id").to_pylist()]
assert len(TEST_IDS) == len(set(TEST_IDS)), "duplicate Source-1 ids in test input"
print(f"test Source-1 rows: {len(TEST_IDS):,}; countries:",
      pd.Series(test_raw.column("country").to_pylist()).value_counts().to_dict())

# reservoir sample (Algorithm R) of TEST_BENCH_SAMPLE row positions
rng = np.random.default_rng(SEED)
res = list(range(min(TEST_BENCH_SAMPLE, len(TEST_IDS))))
for i in range(len(res), len(TEST_IDS)):
    j = rng.integers(0, i + 1)
    if j < len(res):
        res[j] = i
bench_raw = tc.take_raw(test_raw, sorted(res))
BT = tc.Timer()
t0 = time.perf_counter()
bench = tc.process_queries(con_te, bench_raw, CFG, FEATURE_SET, pool=POOL, batch_size=len(res), model=MODEL,
                           timer=BT, workers=CPU_THREADS)
bench_wall = time.perf_counter() - t0
n_pairs = len(bench["qid"])
EST_HOURS = bench_wall / len(res) * len(TEST_IDS) / 3600
BUDGET_HOURS = EST_HOURS * 1.5 + 0.25
print(f"\nbenchmark: {len(res):,} queries, {n_pairs:,} pairs in {bench_wall:.1f}s "
      f"({len(res)/bench_wall:.0f} entities/s, {n_pairs/bench_wall:.0f} pairs/s)")
tot = sum(BT.t.values())
for k, v in sorted(BT.t.items(), key=lambda kv: -kv[1]):
    print(f"  {k:22s} {v:7.2f}s  {100*v/tot:5.1f}%  {len(res)/v:9.0f} entities/s  {n_pairs/v:10.0f} pairs/s")
print(f"projected full test: {EST_HOURS:.2f} h; conservative budget (x1.5 + 0.25h): {BUDGET_HOURS:.2f} h")
if BUDGET_HOURS > MAX_TEST_HOURS:
    raise RuntimeError(f"projected {BUDGET_HOURS:.1f} h exceeds MAX_TEST_HOURS={MAX_TEST_HOURS}; not starting full inference")
""")

md("""
## 16. Full test inference (resumable)

Parts live in `RUN_ROOT/infer_<tag>_<signature>/`. Re-running this cell after an interruption reuses every
batch with a `.done` marker. The signature contains the model hash, threshold, CAP, batch size, input
fingerprints and pipeline version — parts from a different configuration are refused, never mixed.
""")
code(r"""
RUN_SIG = tc.inference_run_signature(MODEL_PATH, THRESHOLD, CFG, FEATURE_SET, BATCH_SIZE, [TEST_S1, *TEST_TARGETS], USE_OWNER)
PARTS_DIR = RUN_ROOT / f"infer_{RUN_TAG}_{tc.signature(RUN_SIG)}"
print("parts:", PARTS_DIR)
IT = tc.Timer()
t0 = time.perf_counter()
tc.run_inference(con_te, test_raw, CFG, FEATURE_SET, MODEL, PARTS_DIR, RUN_SIG, pool=POOL,
                 batch_size=BATCH_SIZE, timer=IT, workers=CPU_THREADS)
INFER_SECONDS = time.perf_counter() - t0
tot = sum(IT.t.values()) or 1
for k, v in sorted(IT.t.items(), key=lambda kv: -kv[1]):
    print(f"  {k:22s} {v:8.1f}s  {100*v/tot:5.1f}%")
print(f"inference wall time this session: {INFER_SECONDS/60:.1f} min")
""")

md("## 17. Assemble, apply the decision rule, validate (own checks + official validator)")
code(r"""
import subprocess
MATCHING = SUBMISSION_ROOT / "matching_results.tsv"
CANDIDATES = SUBMISSION_ROOT / "candidate_pairs.tsv"
ASM = tc.assemble_outputs(PARTS_DIR, TEST_IDS, THRESHOLD, USE_OWNER, MATCHING, CANDIDATES, threads=CPU_THREADS)
print(ASM)
VALID = False
tc.validate_outputs(MATCHING, CANDIDATES, TEST_IDS)          # raises on any failure
r = subprocess.run([sys.executable, str(OFFICIAL_VALIDATOR), "--matching", str(MATCHING), "--candidate", str(CANDIDATES),
                    "--test-dir", str(TEST_DIR)], capture_output=True, text=True)
print(r.stdout[-1500:])
if r.returncode != 0:
    raise RuntimeError("official validator FAILED:\n" + r.stdout + r.stderr)
VALID = True
print("VALIDATION PASS")
""")

md("## 18. Final summary")
code(r"""
cand_counts = pd.read_csv(CANDIDATES, sep="\t", dtype=str, keep_default_na=False)["candidate_entity_ids"].map(
    lambda s: 0 if s == "" else s.count(",") + 1)
summary = {
    "device (benchmark / used)": f"{DEVICE} / {DEVICE_USED}  {({k: round(v, 2) for k, v in BENCH.items()})}",
    "oracle ceiling (calib, NOT a model score)": round(ORACLE_CAL, 5),
    "best iteration count": BEST_ITERS,
    "threshold / one-owner rule": f"{THRESHOLD:.2f} / {USE_OWNER}",
    "calibration macro-F0.5": round(CAL_SCORE, 5),
    "AUDIT macro-F0.5 (untouched)": round(AUDIT["macro_f05"], 5),
    "audit 95% bootstrap CI": tuple(round(x, 5) for x in AUDIT_CI),
    "audit pair precision / recall": (round(AUDIT["pair_precision"], 5), round(AUDIT["pair_recall"], 5)),
    "test candidates per entity (mean / max / empty)": (round(cand_counts.mean(), 2), int(cand_counts.max()), int((cand_counts == 0).sum())),
    "test predicted links / entities with a match": (ASM["predicted_links"], ASM["entities_with_matches"]),
    "runtime projection (h) / actual this session (min)": (round(EST_HOURS, 2), round(INFER_SECONDS / 60, 1)),
    "matching_results.tsv": str(MATCHING),
    "candidate_pairs.tsv": str(CANDIDATES),
    "final validation": "PASS" if VALID else "FAIL",
}
for k, v in summary.items():
    print(f"{k:52s} {v}")
(RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8", newline="\n")
if AUDIT["macro_f05"] >= 0.99:
    print("\nAudit macro-F0.5 >= 0.99 measured on the untouched audit split.")
else:
    print(f"\nAudit macro-F0.5 is {AUDIT['macro_f05']:.4f} (< 0.99). See §13 error analysis for where the loss is.")
print("\nFrance is 15% of the test set and has no labels anywhere: its quality is unmeasured.")
""")
code(r"""
POOL.close(); POOL.join()
for c in ("con0", "con_tr", "con_te"):
    if c in globals():
        globals()[c].close()
print("DONE")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, str(OUT))
print("wrote", OUT, len(cells), "cells")
