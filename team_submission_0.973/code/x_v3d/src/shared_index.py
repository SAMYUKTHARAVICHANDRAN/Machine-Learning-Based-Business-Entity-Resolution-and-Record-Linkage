"""Disk-backed blocking index that many worker processes can share.

Windows has no fork(), so worker processes cannot inherit an in-memory index. Each
country's index is written once as .npy arrays and every worker np.load()s them with
mmap_mode="r": the OS page cache holds a single physical copy for all workers.

Retrieval routes (each candidate records which routes found it):
  name   char-3gram TF-IDF top-K     (common n-grams pruned by max_df)
  addr   word uni+bigram TF-IDF top-K
  num    address numbers of >= 3 digits, postings capped at MAX_POSTINGS
  nkey   (v3) exact normalized-name key: sorted name tokens; groups <= NKEY_MAX_GROUP, the
         NKEY_TAKE best by address cosine are kept for large groups
  akey   (v3) (house number, street word) pairs from the same address component, any number
         length (French/Indian numbers are mostly 1-2 digits), postings <= AKEY_MAX_POSTINGS

v2 also stores, per record, the normalized name/address (computed once here instead of
once per candidate pair) and corpus statistics mined from the catalogue itself:
  name_freq   how many records in the country share this normalized name
  token idf   rarity of each name token in the country
"""
import hashlib
import math
import os
import pickle
import re
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from norm import norm_name, norm_addr, nums, base_norm

MIN_NUM_LEN = 3
MAX_POSTINGS = 50
ROUTE_NAME, ROUTE_ADDR, ROUTE_NUM, ROUTE_NKEY, ROUTE_AKEY, ROUTE_NWORD = 1, 2, 4, 8, 16, 32
NWORD_TOPK = 30
NKEY_MAX_GROUP, NKEY_TAKE = 1000, 60
AKEY_MAX_POSTINGS, AKEY_TAKE = 200, 60
STOP_ADDR_WORDS = {"road", "street", "avenue", "drive", "lane", "nagar", "floor", "near", "opposite", "unit",
                   "suite", "block", "sector", "colony", "north", "south", "east", "west", "main", "cross",
                   "phase", "plot", "house", "door", "null", "city", "district", "village", "residence",
                   "boulevard", "route", "place", "allee", "impasse", "chemin", "cours", "square"}
WORD_RE = re.compile(r"[a-z]{4,}")
DIGIT_RE = re.compile(r"\d+")


def name_key(nn):
    return " ".join(sorted(set(nn.split())))


def addr_keys(raw_addr):
    """(number, street word) pairs within one comma-separated component of the address."""
    out = set()
    for comp in base_norm(raw_addr).split(","):
        ns = {d.lstrip("0") or "0" for d in DIGIT_RE.findall(comp)}
        if not ns:
            continue
        ws = {w for w in WORD_RE.findall(comp) if w not in STOP_ADDR_WORDS}
        out.update(f"{n}|{w}" for n in ns for w in ws)
    return out


def _save_postings(path, prefix, post, max_len):
    keys, lists = [], []
    for tok, ids in post.items():
        if len(ids) <= max_len:
            keys.append(hkey(tok))
            lists.append(ids)
    keys_arr = np.array(keys, dtype=np.uint64)
    order = np.argsort(keys_arr, kind="stable")
    ptr = np.zeros(len(order) + 1, dtype=np.int64)
    np.cumsum([len(lists[k]) for k in order], out=ptr[1:])
    flat = np.fromiter((i for k in order for i in lists[k]), dtype=np.int32, count=int(ptr[-1]))
    np.save(os.path.join(path, f"{prefix}_keys.npy"), keys_arr[order])
    np.save(os.path.join(path, f"{prefix}_ptr.npy"), ptr)
    np.save(os.path.join(path, f"{prefix}_post.npy"), flat)


class _Postings:
    def __init__(self, path, prefix):
        self.keys = np.load(os.path.join(path, f"{prefix}_keys.npy"), mmap_mode="r")
        self.ptr = np.load(os.path.join(path, f"{prefix}_ptr.npy"), mmap_mode="r")
        self.post = np.load(os.path.join(path, f"{prefix}_post.npy"), mmap_mode="r")

    def get(self, tok):
        k = np.uint64(hkey(tok))
        j = int(np.searchsorted(self.keys, k))
        if j < len(self.keys) and self.keys[j] == k:
            return self.post[self.ptr[j]:self.ptr[j + 1]]
        return ()
INDIC = re.compile(r"[ऀ-෿]")


def hkey(text):
    """Process-stable 64-bit key (the builtin hash is salted per process)."""
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def _save_csr(path, prefix, m):
    np.save(os.path.join(path, f"{prefix}_data.npy"), m.data.astype(np.float32))
    np.save(os.path.join(path, f"{prefix}_indices.npy"), m.indices.astype(np.int32))
    np.save(os.path.join(path, f"{prefix}_indptr.npy"), m.indptr.astype(np.int64))


def _load_csr(path, prefix, shape):
    data = np.load(os.path.join(path, f"{prefix}_data.npy"), mmap_mode="r")
    indices = np.load(os.path.join(path, f"{prefix}_indices.npy"), mmap_mode="r")
    indptr = np.load(os.path.join(path, f"{prefix}_indptr.npy"), mmap_mode="r")
    return sp.csr_matrix((data, indices, indptr), shape=shape, copy=False)


def _save_table(path, prefix, counter):
    keys = np.fromiter((hkey(k) for k in counter), dtype=np.uint64, count=len(counter))
    vals = np.fromiter(counter.values(), dtype=np.int32, count=len(counter))
    order = np.argsort(keys, kind="stable")
    np.save(os.path.join(path, f"{prefix}_keys.npy"), keys[order])
    np.save(os.path.join(path, f"{prefix}_vals.npy"), vals[order])


class _Table:
    """Sorted-hash lookup table, memory-mapped."""

    def __init__(self, path, prefix):
        self.keys = np.load(os.path.join(path, f"{prefix}_keys.npy"), mmap_mode="r")
        self.vals = np.load(os.path.join(path, f"{prefix}_vals.npy"), mmap_mode="r")

    def get(self, text):
        k = np.uint64(hkey(text))
        j = int(np.searchsorted(self.keys, k))
        return int(self.vals[j]) if j < len(self.keys) and self.keys[j] == k else 0


def build(records, out_dir, max_features=300_000, log=print):
    """Build and save one index per country label found in `records`."""
    os.makedirs(out_dir, exist_ok=True)
    by_country = defaultdict(list)
    for r in records:
        by_country[r[3]].append(r)

    for country, rows in by_country.items():
        cdir = os.path.join(out_dir, country)
        os.makedirs(cdir, exist_ok=True)
        big = len(rows) > 1000
        n = len(rows)
        log(f"  [{country}] {n:,} records")

        nnames = [norm_name(r[1]) or base_norm(r[1]).strip() for r in rows]
        naddrs = [norm_addr(r[2], r[3]) for r in rows]
        meta = {"country": country, "n": n}

        for prefix, texts, kw in (
            ("name", nnames, dict(analyzer="char_wb", ngram_range=(3, 3))),
            # v3b: word uni+bigram names. Char 3-grams of generic names are all pruned by max_df
            # (4.8% of India and 7.8% of France names had no feature); word bigrams stay specific.
            ("nword", nnames, dict(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")),
            ("addr", naddrs, dict(analyzer="word", ngram_range=(1, 2),
                                  token_pattern=r"(?u)\b\w+\b")),
        ):
            # max_df drops n-grams shared by >1% of docs; without it the batch product
            # is effectively dense (profiled: 2-grams overlapped 83% of the index)
            vec = TfidfVectorizer(min_df=2, max_df=0.01 if big else 1.0,
                                  max_features=max_features, dtype=np.float32, **kw)
            m = vec.fit_transform(texts)
            empty = float(np.mean(np.diff(m.indptr) == 0)); max_df = vec.max_df
            log(f"  [{country}] {prefix}: max_df={max_df} vocab={m.shape[1]:,} empty={empty:.2%}")
            vec.stop_words_ = None          # can hold millions of pruned terms; unused
            with open(os.path.join(cdir, f"{prefix}_vec.pkl"), "wb") as f:
                pickle.dump(vec, f, protocol=pickle.HIGHEST_PROTOCOL)
            mt = sp.csr_matrix(m.T)         # (V, N): batch product is CSR @ CSR
            _save_csr(cdir, f"{prefix}_MT", mt)
            meta[f"{prefix}_V"] = mt.shape[0]
            del m, mt

        # corpus statistics
        name_freq = Counter(nnames)
        tok_df = Counter(t for nn in nnames for t in set(nn.split()))
        _save_table(cdir, "namefreq", name_freq)
        _save_table(cdir, "tokdf", tok_df)
        idf = lambda t: math.log((n + 1) / (tok_df.get(t, 0) + 1))

        # per-record text: raw fields, normalized fields, script flag, name freq, token idfs
        off = np.zeros(n + 1, dtype=np.int64)
        with open(os.path.join(cdir, "text_blob.bin"), "wb") as f:
            for i, (r, nn, na) in enumerate(zip(rows, nnames, naddrs)):
                idfs = " ".join(f"{idf(t):.2f}" for t in nn.split())
                b = (f"{r[0]}\t{r[1]}\t{r[2]}\t{nn}\t{na}\t{1 if INDIC.search(r[1]) else 0}"
                     f"\t{name_freq[nn]}\t{idfs}").encode("utf-8")
                f.write(b)
                off[i + 1] = off[i] + len(b)
        np.save(os.path.join(cdir, "text_off.npy"), off)

        post = defaultdict(list)
        for i, r in enumerate(rows):
            for tok in nums(r[2]):
                if len(tok) >= MIN_NUM_LEN:
                    post[tok].append(i)
        keys, lists = [], []
        for tok, ids in post.items():
            if len(ids) <= MAX_POSTINGS:
                keys.append(hkey(tok))
                lists.append(ids)
        keys_arr = np.array(keys, dtype=np.uint64)
        order = np.argsort(keys_arr, kind="stable")
        ptr = np.zeros(len(order) + 1, dtype=np.int64)
        np.cumsum([len(lists[k]) for k in order], out=ptr[1:])
        flat = np.fromiter((i for k in order for i in lists[k]),
                           dtype=np.int32, count=int(ptr[-1]))
        np.save(os.path.join(cdir, "num_keys.npy"), keys_arr[order])
        np.save(os.path.join(cdir, "num_ptr.npy"), ptr)
        np.save(os.path.join(cdir, "num_post.npy"), flat)
        nnames_keep = nnames
        del post, keys, lists, naddrs

        npost, apost = defaultdict(list), defaultdict(list)
        for i, (r, nn) in enumerate(zip(rows, nnames_keep)):
            if nn:
                npost[name_key(nn)].append(i)
                npost["g|" + nn.replace(" ", "")].append(i)     # v3d: space-free key (domain forms)
            for k in addr_keys(r[2]):
                apost[k].append(i)
        _save_postings(cdir, "nkey", npost, NKEY_MAX_GROUP)
        _save_postings(cdir, "akey", apost, AKEY_MAX_POSTINGS)
        del npost, apost, nnames_keep

        with open(os.path.join(cdir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        log(f"  [{country}] saved")


class CountryIndex:
    def __init__(self, cdir):
        with open(os.path.join(cdir, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self.country, self.n = meta["country"], meta["n"]
        with open(os.path.join(cdir, "name_vec.pkl"), "rb") as f:
            self.name_vec = pickle.load(f)
        with open(os.path.join(cdir, "addr_vec.pkl"), "rb") as f:
            self.addr_vec = pickle.load(f)
        self.name_mt = _load_csr(cdir, "name_MT", (meta["name_V"], self.n))
        with open(os.path.join(cdir, "nword_vec.pkl"), "rb") as f:
            self.nword_vec = pickle.load(f)
        self.nword_mt = _load_csr(cdir, "nword_MT", (meta["nword_V"], self.n))
        self.addr_mt = _load_csr(cdir, "addr_MT", (meta["addr_V"], self.n))
        self.off = np.load(os.path.join(cdir, "text_off.npy"), mmap_mode="r")
        self.blob = np.memmap(os.path.join(cdir, "text_blob.bin"), dtype=np.uint8, mode="r")
        self.num_keys = np.load(os.path.join(cdir, "num_keys.npy"), mmap_mode="r")
        self.num_ptr = np.load(os.path.join(cdir, "num_ptr.npy"), mmap_mode="r")
        self.num_post = np.load(os.path.join(cdir, "num_post.npy"), mmap_mode="r")
        self.nkey = _Postings(cdir, "nkey")
        self.akey = _Postings(cdir, "akey")
        self.namefreq = _Table(cdir, "namefreq")
        self.tokdf = _Table(cdir, "tokdf")

    def record(self, i):
        """(id, raw_name, raw_addr, norm_name, norm_addr, is_indic, name_freq, idfs)"""
        f = bytes(self.blob[self.off[i]:self.off[i + 1]]).decode("utf-8").split("\t")
        return (f[0], f[1], f[2], f[3], f[4], f[5] == "1", int(f[6]),
                [float(x) for x in f[7].split()] if f[7] else [])

    def idf(self, token):
        return math.log((self.n + 1) / (self.tokdf.get(token) + 1))

    def postings(self, tok):
        k = np.uint64(hkey(tok))
        j = int(np.searchsorted(self.num_keys, k))
        if j < len(self.num_keys) and self.num_keys[j] == k:
            return self.num_post[self.num_ptr[j]:self.num_ptr[j + 1]]
        return ()


def _topk(S, bi, k):
    lo, hi = S.indptr[bi], S.indptr[bi + 1]
    idx, dat = S.indices[lo:hi], S.data[lo:hi]
    if len(idx) > k:
        idx = idx[np.argpartition(-dat, k)[:k]]
    return idx


def _lookup(S, bi, cl):
    """Values of row bi of sorted-index CSR S at columns cl (0 where absent)."""
    lo, hi = S.indptr[bi], S.indptr[bi + 1]
    ridx, rdat = S.indices[lo:hi], S.data[lo:hi]
    out = np.zeros(len(cl), dtype=np.float32)
    if len(ridx):
        j = np.searchsorted(ridx, cl)
        ok = j < len(ridx)
        ok[ok] = ridx[j[ok]] == cl[ok]
        out[ok] = rdat[j[ok]]
    return out


class SharedBlocker:
    """Read-only, memory-mapped blocker. Unknown country labels get no candidates."""

    def __init__(self, index_dir, topk=50, addr_topk=50):
        self.topk, self.addr_topk = topk, addr_topk
        self.index = {}
        for c in os.listdir(index_dir):
            cdir = os.path.join(index_dir, c)
            if os.path.isfile(os.path.join(cdir, "meta.pkl")):
                ci = CountryIndex(cdir)
                self.index[ci.country] = ci

    def candidates_batch(self, queries, qnorm):
        """queries: raw S1 tuples; qnorm: matching list of (norm_name, norm_addr).

        Returns one entry per query: None for an unknown country, else
        (country_index, target_row_ids, name_cos, addr_cos, route_bits)."""
        out = [None] * len(queries)
        by_country = {}
        for pos, q in enumerate(queries):
            by_country.setdefault(q[3], []).append(pos)

        for country, positions in by_country.items():
            ci = self.index.get(country)
            if ci is None:
                continue
            Sn = (ci.name_vec.transform([qnorm[p][0] for p in positions]) @ ci.name_mt).tocsr()
            Sa = (ci.addr_vec.transform([qnorm[p][1] for p in positions]) @ ci.addr_mt).tocsr()
            Sw = (ci.nword_vec.transform([qnorm[p][0] for p in positions]) @ ci.nword_mt).tocsr()
            Sn.sort_indices()
            Sa.sort_indices()
            Sw.sort_indices()

            for bi, pos in enumerate(positions):
                q = queries[pos]
                bits = defaultdict(int)
                for i in _topk(Sn, bi, self.topk).tolist():
                    bits[i] |= ROUTE_NAME
                for i in _topk(Sa, bi, self.addr_topk).tolist():
                    bits[i] |= ROUTE_ADDR
                for i in _topk(Sw, bi, NWORD_TOPK).tolist():
                    bits[i] |= ROUTE_NWORD
                for tok in nums(q[2]):
                    if len(tok) >= MIN_NUM_LEN:
                        for i in ci.postings(tok):
                            bits[int(i)] |= ROUTE_NUM
                # v3 key routes; large groups keep the entries closest in address (or name) space
                nk = name_key(qnorm[pos][0]) if qnorm[pos][0] else ""
                grp = np.asarray(ci.nkey.get(nk), dtype=np.int64) if nk else np.zeros(0, np.int64)
                if nk:
                    g2 = np.asarray(ci.nkey.get("g|" + qnorm[pos][0].replace(" ", "")), dtype=np.int64)
                    if len(g2):
                        grp = np.union1d(grp, g2)
                if len(grp) > NKEY_TAKE:
                    grp = grp[np.argsort(-_lookup(Sa, bi, grp), kind="stable")[:NKEY_TAKE]]
                for i in grp.tolist():
                    bits[i] |= ROUTE_NKEY
                ak = set()
                for k in addr_keys(q[2]):
                    ak.update(int(i) for i in ci.akey.get(k))
                ak = np.fromiter(ak, dtype=np.int64, count=len(ak))
                if len(ak) > AKEY_TAKE:
                    ak = ak[np.argsort(-_lookup(Sn, bi, ak), kind="stable")[:AKEY_TAKE]]
                for i in ak.tolist():
                    bits[i] |= ROUTE_AKEY
                cl = np.fromiter(bits.keys(), dtype=np.int64, count=len(bits))
                rb = np.fromiter(bits.values(), dtype=np.int16, count=len(bits))
                out[pos] = (ci, cl, _lookup(Sn, bi, cl), _lookup(Sa, bi, cl), rb, _lookup(Sw, bi, cl))
        return out
