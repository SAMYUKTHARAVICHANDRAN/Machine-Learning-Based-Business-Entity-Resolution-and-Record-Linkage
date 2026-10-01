"""Does TRIDENT add anything to X's v3d? Measured on 20k labelled train S1 that neither model trained on.
Decision rule for every system: X's per-entity expected-F0.5 selection (decide.select_expected_f)."""
import os, sys, json, time, hashlib, random, pickle, multiprocessing as mp
from pathlib import Path
import numpy as np

ROOT = Path(r"E:\Datasets\Amazon_ML_Challenge")
XSRC = Path(r"C:\Users\krith\Downloads\v3d_full_code\v3d_full_code\code\business_entity_resolution")
D = ROOT / "student_resource" / "dataset" / "train"
WORK = ROOT / "work" / "merge"
XIDX = WORK / "x_idx_train"
sys.path.insert(0, str(XSRC / "src"))
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def xbucket(eid):
    return int.from_bytes(hashlib.blake2b(eid.encode(), digest_size=4).digest(), "little") % 100


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    import pandas as pd, duckdb
    import trident_core as tc
    from decide import select_expected_f
    WORK.mkdir(parents=True, exist_ok=True)
    s1df = pd.read_csv(D / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    gtdf = pd.read_csv(D / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = {s: set(m.split(",")) if m else set() for s, m in zip(gtdf.source1_entity_id, gtdf.matched_entity_ids)}

    # ---- 1. X's train index + S1 name counts (label-free)
    if not any((XIDX / c / "meta.pkl").exists() for c in (os.listdir(XIDX) if XIDX.exists() else [])):
        from shared_index import build
        recs = []
        for f in ("train_source2.tsv", "train_source3.tsv"):
            df = pd.read_csv(D / f, sep="\t", dtype=str, keep_default_na=False)
            recs += list(zip(df.entity_id, df.business_name, df.business_address, df.country))
        t = time.perf_counter(); build(recs, str(XIDX), log=log); del recs
        log(f"X index built in {time.perf_counter() - t:.0f}s")
    if not (XIDX / "s1_namecount.pkl").exists():
        from collections import Counter
        from norm import norm_name, base_norm
        from shared_index import name_key
        c = Counter(cn + "|" + name_key(norm_name(n) or base_norm(n).strip()) for cn, n in zip(s1df.country, s1df.business_name))
        pickle.dump(dict(c), open(XIDX / "s1_namecount.pkl", "wb"))

    # ---- 2. evaluation entities: X holdout region (bucket >= 90, never in X's fit) and not in TRIDENT training buckets
    ids = s1df.entity_id.tolist()
    cand = [e for e in ids if xbucket(e) >= 90 and not (100 <= tc.digest_id("split:" + e) % 10000 < 400)]
    EV = sorted(random.Random(2026).sample(cand, 20000))
    rows = s1df.set_index("entity_id").loc[EV]
    queries = list(zip(EV, rows.business_name, rows.business_address, rows.country))

    # ---- 3a. X scores
    xp = WORK / "x_scores.pkl"
    if xp.exists():
        XS = pickle.load(open(xp, "rb"))
    else:
        import pairs
        XS = pairs.score(queries, str(XIDX), str(XSRC / "models" / "model_v3d.pkl"), log=log)
        pickle.dump(XS, open(xp, "wb"))
    # ---- 3b. TRIDENT scores (same model, index and lexicon as the submission)
    tp = WORK / "t_scores.pkl"
    if tp.exists():
        TS = pickle.load(open(tp, "rb"))
    else:
        from catboost import CatBoostClassifier
        RUN = ROOT / "work" / "trident_v2"
        lex = json.loads(next(RUN.glob("lexicon_*.json")).read_text(encoding="utf-8"))
        tc.set_lexicon(lex)
        cfg = dict(norm_mode="canon", routes="v2", country_blocking=True, tok_cap=40, other_cap=400,
                   pre_cap=300, cap=60, prio="v2")
        model = CatBoostClassifier(); model.load_model(str(next((RUN / "models").glob("catboost_ede1*.cbm"))))
        con = duckdb.connect(str(next((RUN / "index").glob("train_3712*.duckdb"))), read_only=True)
        tc.configure_duckdb(con, 20, 12, RUN / "duck_tmp")
        import pyarrow as pa
        raw = pa.table({"entity_id": EV, "business_name": list(rows.business_name),
                        "business_address": list(rows.business_address), "country": list(rows.country)})
        with mp.get_context("spawn").Pool(20, initializer=tc.set_lexicon, initargs=(lex,)) as pool:
            r = tc.process_queries(con, raw, cfg, "v2", pool=pool, batch_size=20000, model=model, log=log)
        TS = {}
        for q, c, p in zip(r["qid"], r["cand_id"], r["p"]):
            TS.setdefault(EV[q], []).append((c, float(p)))
        pickle.dump(TS, open(tp, "wb"))
    log("scored: X pairs", sum(map(len, XS.values())), "| TRIDENT pairs", sum(map(len, TS.values())))

    # ---- 4. union table
    R = []
    for e in EV:
        px, pt = dict(XS.get(e, [])), dict(TS.get(e, []))
        for t in set(px) | set(pt):
            R.append((e, t, px.get(t, np.nan), pt.get(t, np.nan), t in gt[e]))
    df = pd.DataFrame(R, columns=["s1", "t", "px", "pt", "y"])
    ntrue = {e: len(gt[e]) for e in EV}

    def macro(sel):  # sel: {s1: set}
        s = []
        for e in EV:
            P, T = sel.get(e, set()), gt[e]
            tp_ = len(P & T); fp = len(P) - tp_; fn = len(T) - tp_
            s.append((1.0 if not P else 0.0) if not T else (5 * tp_ / (5 * tp_ + 4 * fp + fn) if tp_ else 0.0))
        return float(np.mean(s)), np.array(s)

    def decide(col):
        sel = {}
        for e, g in df[df[col].notna()].groupby("s1"):
            sel[e] = select_expected_f(list(zip(g.t, g[col])), total=float(g[col].sum()))
        return sel

    fx, sx = macro(decide("px")); ft, _ = macro(decide("pt"))
    oracle_x = np.mean([(1.0 if not gt[e] else 5 * len(set(dict(XS.get(e, [])).keys()) & gt[e]) /
                         (5 * len(set(dict(XS.get(e, [])).keys()) & gt[e]) + len(gt[e] - set(dict(XS.get(e, [])).keys()))))
                        if gt[e] or True else 0 for e in EV])
    oracle_u = np.mean([1.0 if not gt[e] else (lambda k: 5 * k / (5 * k + len(gt[e]) - k))(len((set(dict(XS.get(e, []))) | set(dict(TS.get(e, [])))) & gt[e])) for e in EV])
    log(f"X v3d alone      macro-F0.5 {fx:.5f}   (candidate ceiling {oracle_x:.5f})")
    log(f"TRIDENT alone    macro-F0.5 {ft:.5f}")
    log(f"union candidates ceiling {oracle_u:.5f}")

    # ---- 5. cross-fitted combiner (2 folds by entity hash)
    from sklearn.ensemble import HistGradientBoostingClassifier
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    F = np.column_stack([np.nan_to_num(lg(df.px), nan=-14), np.nan_to_num(lg(df.pt), nan=-14),
                         df.px.notna(), df.pt.notna()]).astype(np.float32)
    fold = np.array([tc.digest_id("mfold:" + e) % 2 for e in df.s1])
    df["pc"] = np.nan
    for k in (0, 1):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
        m.fit(F[fold != k], df.y[fold != k])
        df.loc[fold == k, "pc"] = m.predict_proba(F[fold == k])[:, 1]
    fc, sc = macro(decide("pc"))
    d = sc - sx
    rng = np.random.default_rng(2026)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    log(f"COMBINED (cross-fit) macro-F0.5 {fc:.5f}   gain vs X {fc - fx:+.5f}  95% CI [{np.quantile(boots, .025):+.5f}, {np.quantile(boots, .975):+.5f}]")
    json.dump({"x": fx, "trident": ft, "combined": fc, "ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]},
              open(WORK / "merge_result.json", "w"))
