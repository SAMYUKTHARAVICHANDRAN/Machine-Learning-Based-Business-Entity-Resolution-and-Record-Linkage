# -*- coding: utf-8 -*-
"""
TRIDENT entity-resolution core library.

Every stage that must behave identically between training, calibration, audit and
test inference lives here (normalisation, blocking keys, DuckDB retrieval, the
canonical batched feature builder, metrics, output validation). The notebook
imports this module; multiprocessing workers import it too, which is why it is a
real module rather than notebook cells (Windows uses spawn-based workers that
cannot pickle functions defined in a notebook).
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

PIPELINE_VERSION = "trident-2.0.0"

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def digest_id(value: str) -> int:
    """Stable 64-bit hash (process independent, unlike Python's salted hash())."""
    return int.from_bytes(hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest(), "big")


def to_text(value) -> str:
    """Null-safe conversion to str (None / NaN / pandas NA -> '')."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    try:
        import pandas as pd  # noqa: WPS433 (optional)
        if value is pd.NA:
            return ""
    except Exception:
        pass
    return str(value)


def file_fingerprint(path) -> dict:
    p = Path(path)
    st = p.stat()
    return {"path": str(p.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def signature(obj) -> str:
    """Deterministic short hash of a JSON-serialisable object (cache signatures)."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=12).hexdigest()


def file_sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Timer:
    """Accumulating named timers used for the profiling report."""

    def __init__(self):
        self.t = {}

    def add(self, name, seconds):
        self.t[name] = self.t.get(name, 0.0) + seconds

    class _Ctx:
        def __init__(self, parent, name):
            self.parent, self.name = parent, name

        def __enter__(self):
            self.t0 = time.perf_counter()

        def __exit__(self, *exc):
            self.parent.add(self.name, time.perf_counter() - self.t0)

    def __call__(self, name):
        return Timer._Ctx(self, name)


# ---------------------------------------------------------------------------
# 1. Normalisation
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[^\W_]+")


def norm_spec(value) -> str:
    """Specification normalisation: NFKD, casefold, strip combining marks, keep
    unicode alphanumeric runs, collapse whitespace."""
    value = unicodedata.normalize("NFKD", to_text(value)).casefold()
    value = "".join(c for c in value if not unicodedata.combining(c))
    return " ".join(_WORD_RE.findall(value))


# --- Indic transliteration -------------------------------------------------
# The Unicode Indic blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya,
# Tamil, Telugu, Kannada, Malayalam) share the ISCII layout, so one table keyed
# on the offset inside a 0x80 block transliterates all nine scripts.
_INDIC_LO, _INDIC_HI = 0x0900, 0x0D7F
_INDIC_RE = re.compile("[ऀ-ൿ]")
_I_VOWEL = {0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
            0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
            0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"}
_I_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
           0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
           0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
           0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
           0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
           0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r",
           0x5D: "rh", 0x5E: "f", 0x5F: "y", 0x4E: "t"}
_I_MATRA = {0x3A: "e", 0x3B: "e", 0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u",
            0x43: "ri", 0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o",
            0x4A: "o", 0x4B: "o", 0x4C: "au", 0x4F: "e", 0x62: "li", 0x63: "li"}
_I_SIGN = {0x00: "n", 0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n", 0x7A: "n", 0x7B: "n",
           0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k", 0x50: "om"}
_I_VIRAMA = 0x4D
_I_DIGIT0 = 0x66


def translit_indic(text: str) -> str:
    """Rule-based Indic -> Latin transliteration with inherent-vowel handling and
    word-final schwa deletion (so 'राम' -> 'ram', 'लिमिटेड' -> 'limited')."""
    out = []
    pending_a = False  # consonant emitted whose inherent 'a' is not yet written
    for ch in text:
        cp = ord(ch)
        if _INDIC_LO <= cp <= _INDIC_HI:
            off = (cp - _INDIC_LO) % 0x80
            if off in _I_CONS:
                if pending_a:
                    out.append("a")
                out.append(_I_CONS[off])
                pending_a = True
            elif off in _I_MATRA:
                out.append(_I_MATRA[off])
                pending_a = False
            elif off == _I_VIRAMA:
                pending_a = False
            elif off in _I_VOWEL:
                pending_a = False  # consonant followed by independent vowel: drop schwa
                out.append(_I_VOWEL[off])
            elif off in _I_SIGN:
                if pending_a:
                    out.append("a")
                    pending_a = False
                out.append(_I_SIGN[off])
            elif _I_DIGIT0 <= off <= _I_DIGIT0 + 9:
                if pending_a:
                    pending_a = False
                out.append(chr(ord("0") + off - _I_DIGIT0))
            else:  # nukta, avagraha, length marks, dandas ...
                if off in (0x64, 0x65):
                    pending_a = False
                    out.append(" ")
        else:
            pending_a = False  # word boundary -> schwa deletion
            out.append(ch)
    return "".join(out)


_SPECIAL_LETTERS = str.maketrans({"œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae", "ø": "o", "Ø": "o",
                                  "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ı": "i", "°": "o",
                                  "&": " and "})

# Native-script state names observed in the training targets (exact strings) and
# native legal words. Replaced before transliteration.
_NATIVE_ADDR = {
    "महाराष्ट्र": " mh ", "दिल्ली": " delhi ", "उत्तर प्रदेश": " up ", "ಕರ್ನಾಟಕ": " ka ",
    "தமிழ்நாடு": " tn ", "পশ্চিমবঙ্গ": " wb ", "ગુજરાત": " gj ", "తెలంగాణ": " tg ",
    "हरियाणा": " hr ", "राजस्थान": " rj ", "കേരളം": " kl ", "बिहार": " br ",
    "मध्य प्रदेश": " mp ", "ఆంధ్రప్రదేశ్": " ap ", "ਪੰਜਾਬ": " pb ", "ଓଡ଼ିଶା": " od ",
}
_NATIVE_NAME = {
    "प्रा.": " pvt ", "लि.": " ltd ", "एलएलपी": " llp ",
}
for _w in ("लिमिटेड", "లిమిటెడ్", "ಲಿಮಿಟೆಡ್", "லிமிடெட்", "লিমিটেড", "લિમિટેડ", "ലിമിറ്റഡ്", "ଲିମିଟେଡ୍"):
    _NATIVE_NAME[_w] = " ltd "
for _w in ("प्राइवेट", "ప్రైవేట్", "ಪ್ರೈವೇಟ್", "பிரைவேட்", "প্রাইভেট", "પ્રાઇવેટ", "പ്രൈവറ്റ്"):
    _NATIVE_NAME[_w] = " pvt "


def _replace_all(s: str, table: dict) -> str:
    for k, v in table.items():
        if k in s:
            s = s.replace(k, v)
    return s


def norm_base(value, native_table=None) -> str:
    """v2 base normalisation = native-word replacement + Indic transliteration +
    special letters, then exactly the specification normalisation."""
    s = unicodedata.normalize("NFC", to_text(value))
    if not s:
        return ""
    if _INDIC_RE.search(s):
        if native_table:
            s = _replace_all(s, native_table)
        s = translit_indic(s)
    s = s.translate(_SPECIAL_LETTERS)
    return norm_spec(s)


# --- canonical abbreviation maps (short canonical forms) -------------------
NAME_CANON = {
    "private": "pvt", "prvt": "pvt", "pvt": "pvt", "pte": "pvt", "pvtltd": "pvt ltd",
    "limited": "ltd", "ltd": "ltd", "limitd": "ltd",
    "incorporated": "inc", "incorporation": "inc",
    "corporation": "corp", "corpn": "corp",
    "company": "co", "compagnie": "co", "cie": "co", "comp": "co",
    "and": "", "et": "",
    "etablissements": "ets", "ets": "ets", "etablissement": "ets",
    "international": "intl", "intl": "intl",
    "brothers": "bros", "bros": "bros", "freres": "freres",
    "associates": "assoc", "assoc": "assoc",
    "society": "soc", "societe": "ste",
    "department": "dept", "dept": "dept",
    "praivet": "pvt", "elelpi": "llp",
}
ADDR_CANON = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "avenu": "ave",
    "drive": "dr", "court": "ct", "lane": "ln", "boulevard": "blvd", "bd": "blvd", "bvd": "blvd",
    "bld": "blvd", "place": "pl", "circle": "cir", "terrace": "ter", "highway": "hwy",
    "parkway": "pkwy", "pky": "pkwy", "apartment": "apt", "apartments": "apts", "suite": "ste",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "floor": "fl", "flr": "fl", "building": "bldg",
    "number": "no", "near": "nr", "opposite": "opp", "opp": "opp", "district": "dist",
    "square": "sq", "mount": "mt", "route": "rte", "trail": "trl", "point": "pt",
    "rue": "r", "allee": "all", "impasse": "imp", "chemin": "ch", "saint": "st", "sainte": "ste",
    "quai": "qu", "cours": "crs", "village": "vill", "vil": "vill", "taluka": "tal", "taluk": "tal",
    "tehsil": "tal", "post": "po", "house": "h", "hno": "h no",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "jammu and kashmir": "jk", "jammu kashmir": "jk", "puducherry": "py",
    "pondicherry": "py",
}


def _phrase_regex(d):
    keys = sorted(d, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")


_US_RE, _IN_RE = _phrase_regex(US_STATES), _phrase_regex(IN_STATES)

# Stop lists. Spec lists are reproduced verbatim for norm_mode='spec'.
NAME_STOP_SPEC = frozenset("private limited pvt ltd llc inc corp corporation company the and com".split())
ADDR_STOP_SPEC = frozenset("road street avenue near floor plot shop sector block building district nagar".split())
NAME_STOP = frozenset((
    "pvt ltd llc llp inc corp co the and com pllc plc pc lp ltda of dba fka aka ms m s smt sri shri "
    "shree www in net org sarl sas sasu sa eurl sci snc ei ets et de du des la le les l d france "
    "bros assoc"
).split())
ADDR_STOP = frozenset((
    "rd st ave nr near fl plot shop sector block bldg dist nagar no h flat unit apt apts ste dr ct ln "
    "blvd pl cir ter hwy pkwy n s e w ne nw se sw opp c o via po tal tq vill r de du des la le les "
    "l d bis all imp ch rte a b the of and room ground first second"
).split())


# Learned transliteration lexicon (translit token -> English token), fitted on training pairs only
# by learn_translit_lexicon(); applied to names that contained Indic script. Pool workers receive it
# through the pool initializer set_lexicon().
_LEX: dict = {}


def set_lexicon(lex):
    global _LEX
    _LEX = dict(lex or {})


def lexicon_signature() -> str:
    return signature(sorted(_LEX.items())) if _LEX else "none"


def canon_name(value, mode="canon") -> str:
    if mode == "spec":
        return norm_spec(value)
    raw = to_text(value)
    s = norm_base(raw, _NATIVE_NAME)
    indic = bool(_LEX) and bool(_INDIC_RE.search(raw))
    out = []
    for t in s.split():
        if indic:
            t = _LEX.get(t, t)
        m = NAME_CANON.get(t, t)
        if m:
            out.append(m)
    return " ".join(" ".join(out).split())


def learn_translit_lexicon(con, s1_path, target_paths, truth_path, exclude_s1_ids, max_pairs=400_000,
                           min_count=5, min_share=0.6, min_ratio=0.5, log=print):
    """Learn translit-token -> English-token pairs from TRUE training links whose target name is in
    an Indic script. Entities in exclude_s1_ids (calibration + audit) are never used."""
    from rapidfuzz import fuzz
    import pyarrow as pa

    con.register("lex_excl", pa.table({"s1": pa.array(sorted(exclude_s1_ids), pa.string())}))
    tp = str(Path(truth_path)).replace("\\", "/")
    tg = " UNION ALL ".join(f"SELECT entity_id, business_name FROM {tsv_relation(p)}" for p in target_paths)
    rows = con.execute(f"""
        WITH gl AS (SELECT trim(source1_entity_id) AS s1, trim(unnest(string_split(matched_entity_ids, ','))) AS t
                    FROM read_csv('{tp}', delim='	', header=true, quote='', escape='', all_varchar=true)
                    WHERE coalesce(matched_entity_ids, '') <> ''),
             tg AS (SELECT * FROM ({tg}) WHERE regexp_matches(business_name, '[ऀ-ൿ]'))
        SELECT s.business_name, tg.business_name
        FROM gl JOIN tg ON tg.entity_id = gl.t JOIN {tsv_relation(s1_path)} s ON s.entity_id = gl.s1
        WHERE gl.s1 NOT IN (SELECT s1 FROM lex_excl)
        ORDER BY hash(gl.t) LIMIT {int(max_pairs)}""").fetchall()
    con.unregister("lex_excl")
    saved = dict(_LEX)
    set_lexicon({})
    pair_counts, tok_counts = {}, {}
    for s1_name, t_name in rows:
        e_toks = set(canon_name(s1_name).split())
        for t in canon_name(t_name).split():
            if t in e_toks or t.isdigit() or len(t) < 2:
                continue
            tok_counts[t] = tok_counts.get(t, 0) + 1
            best, best_r = None, 0.0
            for e in e_toks:
                r = fuzz.ratio(t, e) / 100
                if r > best_r or (r == best_r and best is not None and e < best):
                    best, best_r = e, r
            if best is not None and best_r >= min_ratio:
                pair_counts[(t, best)] = pair_counts.get((t, best), 0) + 1
    set_lexicon(saved)
    lex = {}
    for (t, e), c in sorted(pair_counts.items()):
        if c >= min_count and c / tok_counts[t] >= min_share and t != e:
            if t not in lex or c > pair_counts[(t, lex[t])]:
                lex[t] = e
    log(f"[lexicon] {len(rows):,} Indic-name training links -> {len(lex):,} entries "
        f"(e.g. {dict(list(sorted(lex.items(), key=lambda kv: -pair_counts[kv]))[:8])})")
    return lex


def canon_addr(value, country="", mode="canon") -> str:
    if mode == "spec":
        return norm_spec(value)
    s = norm_base(value, _NATIVE_ADDR)
    c = to_text(country).casefold()
    if c in ("us", "usa", "united states"):
        s = _US_RE.sub(lambda m: US_STATES[m.group(1)], s)
    elif c == "india":
        s = _IN_RE.sub(lambda m: IN_STATES[m.group(1)], s)
    out = []
    for t in s.split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
            out.append(t)
            continue
        m = ADDR_CANON.get(t, t)
        if m:
            out.append(m)
    return " ".join(" ".join(out).split())


def name_core_tokens(name_c: str, mode="canon"):
    stop = NAME_STOP_SPEC if mode == "spec" else NAME_STOP
    return [t for t in name_c.split() if t not in stop]


def addr_numbers(addr_c: str):
    """All pure ASCII digit tokens of a normalised address, in order."""
    return [t for t in addr_c.split() if t.isascii() and t.isdigit()]


def _strongest(tokens, stop, k=4, min_len=2):
    seen, cand = set(), []
    for t in tokens:
        if t in seen or t in stop or len(t) < min_len or t.isdigit():
            continue
        seen.add(t)
        cand.append(t)
    cand.sort(key=lambda t: (-len(t), t))
    return cand[:k]


def blocking_fields(name_c: str, addr_c: str, mode="canon"):
    """Per-record blocking material (lists are consumed by DuckDB key SQL)."""
    if mode == "spec":
        ntoks = _strongest(name_c.split(), NAME_STOP_SPEC)
        atoks = _strongest(addr_c.split(), ADDR_STOP_SPEC)
        nums = list(dict.fromkeys(re.findall(r"\b[0-9]{2,}\b", addr_c)))[:4]
        npre4 = name_c.replace(" ", "")[:4]
        core = ""
    else:
        ntoks = _strongest(name_c.split(), NAME_STOP)
        atoks = _strongest(addr_c.split(), ADDR_STOP)
        nums = list(dict.fromkeys(t for t in addr_numbers(addr_c) if len(t) >= 2))[:4]
        core_toks = name_core_tokens(name_c)
        core = "".join(core_toks)
        npre4 = core[:4] if core else name_c.replace(" ", "")[:4]
    return ntoks, atoks, nums, npre4, core


def normalize_chunk(args):
    """Worker: normalise a chunk of raw records.

    args = (entity_ids, names, addrs, countries, mode). Returns a dict of columns.
    """
    ids, names, addrs, countries, mode = args
    out = {k: [] for k in ("entity_id", "country", "name", "addr", "ntoks", "atoks", "nums",
                           "npre4", "core", "name_indic", "addr_indic")}
    for eid, nm, ad, co in zip(ids, names, addrs, countries):
        nm, ad, co = to_text(nm), to_text(ad), to_text(co).strip()
        name_c = canon_name(nm, mode)
        addr_c = canon_addr(ad, co, mode)
        ntoks, atoks, nums, npre4, core = blocking_fields(name_c, addr_c, mode)
        out["entity_id"].append(to_text(eid).strip())
        out["country"].append(co)
        out["name"].append(name_c)
        out["addr"].append(addr_c)
        out["ntoks"].append(ntoks)
        out["atoks"].append(atoks)
        out["nums"].append(nums)
        out["npre4"].append(npre4)
        out["core"].append(core)
        out["name_indic"].append(bool(_INDIC_RE.search(nm)))
        out["addr_indic"].append(bool(_INDIC_RE.search(ad)))
    return out


def normalize_records(ids, names, addrs, countries, mode, pool=None, chunk=50_000):
    """Normalise records (optionally in a process pool) -> pyarrow.Table."""
    import pyarrow as pa
    n = len(ids)
    jobs = [(ids[i:i + chunk], names[i:i + chunk], addrs[i:i + chunk], countries[i:i + chunk], mode)
            for i in range(0, n, chunk)]
    if pool is not None and len(jobs) > 1:
        parts = pool.map(normalize_chunk, jobs)
    else:
        parts = [normalize_chunk(j) for j in jobs]
    cols = {}
    for k in ("entity_id", "country", "name", "addr", "ntoks", "atoks", "nums", "npre4", "core",
              "name_indic", "addr_indic"):
        vals = []
        for p in parts:
            vals.extend(p[k])
        cols[k] = vals
    schema = pa.schema([
        ("entity_id", pa.string()), ("country", pa.string()), ("name", pa.string()),
        ("addr", pa.string()), ("ntoks", pa.list_(pa.string())), ("atoks", pa.list_(pa.string())),
        ("nums", pa.list_(pa.string())), ("npre4", pa.string()), ("core", pa.string()),
        ("name_indic", pa.bool_()), ("addr_indic", pa.bool_()),
    ])
    return pa.table(cols, schema=schema)


# ---------------------------------------------------------------------------
# 2. TSV reading
# ---------------------------------------------------------------------------

def fetch_arrow(result):
    """DuckDB result -> pyarrow.Table (DuckDB >=1.4 returns a RecordBatchReader from .arrow())."""
    out = result.arrow()
    return out.read_all() if hasattr(out, "read_all") else out


def tsv_relation(path) -> str:
    """DuckDB read_csv expression for the challenge TSVs (no quoting, all text)."""
    p = str(Path(path)).replace("\\", "/").replace("'", "''")
    return (f"read_csv('{p}', delim='\t', header=true, quote='', escape='', all_varchar=true, "
            f"columns={{'entity_id':'VARCHAR','business_name':'VARCHAR','business_address':'VARCHAR','country':'VARCHAR'}})")


def read_source_arrow(con, path):
    return fetch_arrow(con.execute(f"SELECT entity_id, business_name, business_address, country FROM {tsv_relation(path)}"))


def read_truth(con, path) -> dict:
    p = str(Path(path)).replace("\\", "/").replace("'", "''")
    rows = con.execute(
        f"SELECT source1_entity_id, coalesce(matched_entity_ids,'') FROM read_csv('{p}', delim='\t', header=true, "
        f"quote='', escape='', all_varchar=true)").fetchall()
    truth = {}
    for s1, m in rows:
        truth[s1.strip()] = frozenset(x.strip() for x in m.split(",") if x.strip())
    return truth


# ---------------------------------------------------------------------------
# 3. Blocking keys + DuckDB index
# ---------------------------------------------------------------------------
# route codes: 1 N exact name, 2 A exact address, 3 n name token, 4 a address token,
# 5 P name-token pair, 6 Q address-token pair, 7 X name-prefix|number,
# 8 C concatenated core name (v2), 9 Y number|address token (v2)
ROUTE_WEIGHT = {1: 8, 2: 8, 3: 1, 4: 1, 5: 3, 6: 3, 7: 3, 8: 6, 9: 3}
NAME_ROUTES = (1, 3, 5, 7, 8)
ADDR_ROUTES = (2, 4, 6, 9)


def key_sql(rel: str, id_col: str, cfg: dict) -> str:
    """SQL producing DISTINCT (id, key, route, w) for relation `rel`.

    Used for BOTH the target index and every query batch, so both sides always
    generate keys with identical logic.
    """
    ctry = "coalesce(country,'')" if cfg["country_blocking"] else "''"
    parts = [
        f"SELECT {id_col} AS id, {ctry} AS c, 'N:'||name AS k, 1 AS r FROM {rel} WHERE name <> ''",
        f"SELECT {id_col}, {ctry}, 'A:'||addr, 2 FROM {rel} WHERE addr <> ''",
        f"SELECT id, c, 'n:'||t, 3 FROM (SELECT {id_col} AS id, {ctry} AS c, unnest(ntoks) AS t FROM {rel})",
        f"SELECT id, c, 'a:'||t, 4 FROM (SELECT {id_col} AS id, {ctry} AS c, unnest(atoks) AS t FROM {rel})",
        f"SELECT id, c, 'P:'||t1||'|'||t2, 5 FROM (SELECT id, c, t1, unnest(l) AS t2 FROM "
        f"(SELECT {id_col} AS id, {ctry} AS c, ntoks AS l, unnest(ntoks) AS t1 FROM {rel})) WHERE t1 < t2",
        f"SELECT id, c, 'Q:'||t1||'|'||t2, 6 FROM (SELECT id, c, t1, unnest(l) AS t2 FROM "
        f"(SELECT {id_col} AS id, {ctry} AS c, atoks AS l, unnest(atoks) AS t1 FROM {rel})) WHERE t1 < t2",
        f"SELECT id, c, 'X:'||p||'|'||t, 7 FROM (SELECT {id_col} AS id, {ctry} AS c, npre4 AS p, unnest(nums) AS t "
        f"FROM {rel}) WHERE p <> ''",
    ]
    if cfg["routes"] == "v2":
        parts += [
            f"SELECT {id_col}, {ctry}, 'C:'||core, 8 FROM {rel} WHERE length(core) >= 4",
            f"SELECT id, c, 'Y:'||t1||'|'||t2, 9 FROM (SELECT id, c, t1, unnest(l) AS t2 FROM "
            f"(SELECT {id_col} AS id, {ctry} AS c, atoks[1:2] AS l, unnest(nums[1:2]) AS t1 FROM {rel}))",
        ]
    union = "\nUNION ALL\n".join(parts)
    wcase = "CASE r " + " ".join(f"WHEN {k} THEN {v}" for k, v in ROUTE_WEIGHT.items()) + " END"
    return (f"SELECT DISTINCT id, hash(c || chr(31) || k) AS key, r::UTINYINT AS route, "
            f"({wcase})::UTINYINT AS w FROM ({union})")


def configure_duckdb(con, threads, memory_limit_gb, temp_dir):
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit='{max(1, int(memory_limit_gb))}GB'")
    con.execute(f"SET temp_directory='{str(temp_dir).replace(chr(92), '/')}'")
    con.execute("SET preserve_insertion_order=false")


def index_signature(target_paths, cfg) -> dict:
    import duckdb
    return {
        "pipeline": PIPELINE_VERSION,
        "inputs": [file_fingerprint(p) for p in target_paths],
        "norm_mode": cfg["norm_mode"], "routes": cfg["routes"],
        "country_blocking": cfg["country_blocking"], "tok_cap": cfg["tok_cap"],
        "other_cap": cfg["other_cap"], "duckdb": duckdb.__version__, "lexicon": lexicon_signature(),
    }


def build_target_index(db_path, target_paths, cfg, pool, threads, memory_limit_gb, temp_dir, log=print):
    """Build (or reuse) the DuckDB target index: tables `records`, `postings`, `meta`.

    Reuse happens only when the stored signature equals the requested signature.
    """
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    db_path = Path(db_path)
    sig = index_signature(target_paths, cfg)
    sig_s = json.dumps(sig, sort_keys=True)
    if db_path.exists():
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            stored = con.execute("SELECT v FROM meta WHERE k='signature'").fetchone()
            ok = con.execute("SELECT v FROM meta WHERE k='complete'").fetchone()
            con.close()
            if stored and ok and stored[0] == sig_s:
                log(f"[index] reusing {db_path.name} (signature match)")
                return db_path
            log(f"[index] signature mismatch -> rebuilding {db_path.name}")
        except Exception as e:  # corrupt / partial file
            log(f"[index] cannot reuse {db_path.name}: {e!r} -> rebuilding")
        db_path.unlink()
    tmp = db_path.with_suffix(".building.duckdb")
    if tmp.exists():
        tmp.unlink()
    pq_path = db_path.with_suffix(".records.parquet")
    t0 = time.perf_counter()
    reader_con = duckdb.connect()
    configure_duckdb(reader_con, threads, memory_limit_gb, temp_dir)
    writer = None
    rid = 0
    try:
        for path in target_paths:
            arr = read_source_arrow(reader_con, path)
            ids = arr.column("entity_id").to_pylist()
            nms = arr.column("business_name").to_pylist()
            ads = arr.column("business_address").to_pylist()
            cts = arr.column("country").to_pylist()
            del arr
            step = 1_000_000
            for i in range(0, len(ids), step):
                tbl = normalize_records(ids[i:i + step], nms[i:i + step], ads[i:i + step], cts[i:i + step],
                                        cfg["norm_mode"], pool=pool)
                tbl = tbl.add_column(0, "rid", pa.array(np.arange(rid, rid + tbl.num_rows, dtype=np.int32)))
                rid += tbl.num_rows
                if writer is None:
                    writer = pq.ParquetWriter(str(pq_path), tbl.schema, compression="zstd")
                writer.write_table(tbl)
                log(f"[index] normalised {rid:,} target records ({time.perf_counter() - t0:.0f}s)")
            del ids, nms, ads, cts
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
        reader_con.close()
    t1 = time.perf_counter()
    con = duckdb.connect(str(tmp))
    configure_duckdb(con, threads, memory_limit_gb, temp_dir)
    pqs = str(pq_path).replace("\\", "/")
    con.execute(f"CREATE TABLE records AS SELECT * FROM read_parquet('{pqs}')")
    countries = [r[0] for r in con.execute("SELECT DISTINCT coalesce(country,'') FROM records ORDER BY 1").fetchall()]
    con.execute("CREATE TABLE postings (key UBIGINT, rid INTEGER, route UTINYINT, w UTINYINT, f UINTEGER, ev FLOAT)")
    parts = [(c,) for c in countries] if cfg["country_blocking"] else [(None,)]
    for (c,) in parts:
        where = "" if c is None else f"WHERE coalesce(country,'') = '{c.replace(chr(39), chr(39) * 2)}'"
        con.execute(f"CREATE OR REPLACE TEMP TABLE rsub AS SELECT * FROM records {where}")
        con.execute(f"CREATE OR REPLACE TEMP TABLE rawk AS {key_sql('rsub', 'rid', cfg)}")
        con.execute("CREATE OR REPLACE TEMP TABLE kf AS SELECT key, count(*)::UINTEGER AS f FROM rawk GROUP BY key")
        con.execute(f"""
            INSERT INTO postings
            SELECT r.key, r.id, r.route, r.w, kf.f, (r.w / (1 + ln(kf.f)))::FLOAT
            FROM rawk r JOIN kf USING (key)
            WHERE kf.f <= CASE WHEN r.route IN (3, 4) THEN {int(cfg['tok_cap'])} ELSE {int(cfg['other_cap'])} END""")
        n_raw = con.execute("SELECT count(*) FROM rawk").fetchone()[0]
        con.execute("DROP TABLE rawk; DROP TABLE kf; DROP TABLE rsub")
        log(f"[index] country={c!r}: {n_raw:,} raw keys ({time.perf_counter() - t1:.0f}s)")
    stats = con.execute("SELECT route, count(*), avg(f) FROM postings GROUP BY route ORDER BY route").fetchall()
    con.execute("CREATE TABLE meta (k VARCHAR, v VARCHAR)")
    con.execute("INSERT INTO meta VALUES ('signature', ?), ('route_stats', ?), ('complete', '1')",
                [sig_s, json.dumps(stats)])
    con.execute("CHECKPOINT")
    con.close()
    os.replace(tmp, db_path)
    log(f"[index] built {db_path.name}: {rid:,} records; postings by route {stats}; "
        f"normalise {t1 - t0:.0f}s, postings {time.perf_counter() - t1:.0f}s")
    return db_path


# ---------------------------------------------------------------------------
# 4. Candidate retrieval
# ---------------------------------------------------------------------------

def retrieve(con, q_tbl, cfg, keep_rids_table: str | None = None):
    """Batch candidate retrieval for query table `q_tbl` (must contain column qid).

    Returns a pyarrow table ordered by (qid, rank) with evidence/priority columns.
    If keep_rids_table is given, rows are filtered AFTER ranking to targets in it
    (used by the context pass; ranks/counts still reflect the full candidate list).
    """
    con.register("q_in", q_tbl)
    con.execute(f"CREATE OR REPLACE TEMP TABLE q_keys AS {key_sql('q_in', 'qid', cfg)}")
    name_routes = ",".join(map(str, NAME_ROUTES))
    addr_routes = ",".join(map(str, ADDR_ROUTES))
    keep = f"WHERE rid IN (SELECT rid FROM {keep_rids_table})" if keep_rids_table else ""
    # spec priority uses JW(address); 'v2' substitutes JW(name) when either address is empty, so a
    # target without an address is not ranked as if its address disagreed completely.
    addr_term = ("jw_addr" if cfg.get("prio", "spec") == "spec"
                 else "CASE WHEN c_addr = '' OR q_addr_empty THEN jw_name ELSE jw_addr END")
    sql = f"""
    WITH agg AS (
        SELECT k.id AS qid, p.rid,
               SUM(p.w)::DOUBLE AS support,
               SUM(p.ev::DOUBLE) AS evidence,
               SUM(CASE WHEN p.route IN ({name_routes}) THEN p.ev::DOUBLE ELSE 0 END) AS ev_name,
               SUM(CASE WHEN p.route IN ({addr_routes}) THEN p.ev::DOUBLE ELSE 0 END) AS ev_addr,
               COUNT(*)::INTEGER AS n_keys
        FROM q_keys k JOIN postings p ON k.key = p.key
        GROUP BY k.id, p.rid
    ),
    pre AS (
        SELECT * FROM agg
        QUALIFY row_number() OVER (PARTITION BY qid ORDER BY evidence DESC, rid) <= {int(cfg['pre_cap'])}
    ),
    sc AS (
        SELECT pre.*, r.entity_id AS cand_id, r.name AS c_name, r.addr AS c_addr, r.country AS c_country,
               r.name_indic AS c_name_indic, r.addr_indic AS c_addr_indic, (q.addr = '') AS q_addr_empty,
               jaro_winkler_similarity(q.name, r.name) AS jw_name,
               jaro_winkler_similarity(q.addr, r.addr) AS jw_addr
        FROM pre JOIN q_in q ON q.qid = pre.qid JOIN records r ON r.rid = pre.rid
    ),
    pr AS (SELECT *, evidence + 3 * jw_name + 4 * {addr_term} AS priority FROM sc),
    rk AS (
        SELECT *, (row_number() OVER (PARTITION BY qid ORDER BY priority DESC, cand_id) - 1)::INTEGER AS rank
        FROM pr QUALIFY rank < {int(cfg['cap'])}
    ),
    fin AS (
        SELECT *,
               COUNT(*) OVER (PARTITION BY qid)::INTEGER AS cand_count,
               MAX(priority) OVER (PARTITION BY qid) AS best_priority,
               coalesce(nth_value(priority, 2) OVER (PARTITION BY qid ORDER BY rank
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING), priority) AS second_priority,
               coalesce(lead(priority) OVER (PARTITION BY qid ORDER BY rank), priority) AS next_priority,
               (row_number() OVER (PARTITION BY qid, left(cand_id, 3) ORDER BY rank) - 1)::INTEGER AS source_rank
        FROM rk
    )
    SELECT * FROM fin {keep} ORDER BY qid, rank
    """
    out = fetch_arrow(con.execute(sql))
    con.unregister("q_in")
    return out


def make_query_table(norm_tbl, qid_offset=0):
    import pyarrow as pa
    return norm_tbl.add_column(0, "qid", pa.array(np.arange(qid_offset, qid_offset + norm_tbl.num_rows,
                                                            dtype=np.int32)))


# ---------------------------------------------------------------------------
# 5. Features (ONE canonical batched implementation + a scalar reference)
# ---------------------------------------------------------------------------
FEATURES_SPEC29 = [
    "name_ratio", "name_wratio", "name_token_sort", "name_token_set", "name_exact",
    "q_name_missing", "c_name_missing", "name_len_ratio",
    "addr_ratio", "addr_wratio", "addr_token_sort", "addr_token_set", "addr_exact",
    "q_addr_missing", "c_addr_missing", "addr_len_ratio",
    "name_jaccard", "addr_jaccard", "num_jaccard", "num_contradiction", "postal_agree",
    "postal_contradiction", "country_equal", "is_source3", "support", "priority", "rank",
    "cand_count", "gap_from_best",
]
FEATURES_EXT = [
    "core_token_set", "core_concat_ratio", "core_concat_partial", "jw_name", "jw_addr",
    "ev_name", "ev_addr", "n_keys", "evidence", "house_num_equal", "num_near_match",
    "c_name_indic", "c_addr_indic", "gap_to_next", "query_ambiguity", "source_rank",
    "name_tok_count_ratio",
]


def feature_names(feature_set: str):
    if feature_set == "spec29":
        return list(FEATURES_SPEC29)
    if feature_set == "v2":
        return FEATURES_SPEC29 + FEATURES_EXT
    raise ValueError(f"unknown feature_set {feature_set!r}")


def _jacc(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _near_num(qn, cn) -> float:
    if not qn or not cn:
        return float("nan")
    for a in qn:
        for b in cn:
            if a == b:
                return 1.0
            if len(a) >= 2 and len(b) >= 2 and (a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)):
                return 1.0
    return 0.0


def _string_parts(s: str, kind: str, mode: str):
    """Per-string derived material, cached per unique string in feature_matrix."""
    toks = s.split()
    if kind == "name":
        core = name_core_tokens(s, mode)
        return {"set": frozenset(toks), "n": len(toks), "core": " ".join(core), "concat": "".join(core)}
    nums = addr_numbers(s)
    return {"set": frozenset(toks), "nums": frozenset(nums), "first": nums[0] if nums else None,
            "postal": frozenset(x for x in nums if len(x) in (5, 6)), "numlist": nums}


def feature_matrix(q_name, q_addr, q_country, c_name, c_addr, c_country, c_id, cols: dict,
                   feature_set="v2", norm_mode="canon", workers=-1) -> np.ndarray:
    """Canonical batched feature builder used for training, calibration, audit and test.

    q_* / c_* are equal-length sequences of normalised strings (one entry per pair);
    `cols` holds retrieval columns (support, priority, rank, ...), numpy arrays.
    """
    from rapidfuzz import fuzz, process

    n = len(q_name)
    names = feature_names(feature_set)
    X = np.empty((n, len(names)), dtype=np.float32)
    if n == 0:
        return X
    q_name, c_name, q_addr, c_addr = list(q_name), list(c_name), list(q_addr), list(c_addr)

    def fz(a, b, scorer):
        return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float64) / 100.0

    j = 0
    for qa, ca in ((q_name, c_name), (q_addr, c_addr)):
        for scorer in (fuzz.ratio, fuzz.WRatio, fuzz.token_sort_ratio, fuzz.token_set_ratio):
            X[:, j] = fz(qa, ca, scorer)
            j += 1
        la = np.fromiter((len(s) for s in qa), dtype=np.float64, count=n)
        lb = np.fromiter((len(s) for s in ca), dtype=np.float64, count=n)
        X[:, j] = np.fromiter((1.0 if (a != "" and a == b) else 0.0 for a, b in zip(qa, ca)), np.float64, n); j += 1
        X[:, j] = (la == 0); j += 1
        X[:, j] = (lb == 0); j += 1
        X[:, j] = np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1.0); j += 1

    cache_n, cache_a = {}, {}

    def pn(s):
        v = cache_n.get(s)
        if v is None:
            v = cache_n[s] = _string_parts(s, "name", norm_mode)
        return v

    def pa_(s):
        v = cache_a.get(s)
        if v is None:
            v = cache_a[s] = _string_parts(s, "addr", norm_mode)
        return v

    ext = feature_set == "v2"
    tok = np.empty((n, 9 if ext else 6), dtype=np.float64)
    core_q, core_c, cc_q, cc_c = ([], [], [], []) if ext else (None, None, None, None)
    for i in range(n):
        a, b = pn(q_name[i]), pn(c_name[i])
        x, y = pa_(q_addr[i]), pa_(c_addr[i])
        tok[i, 0] = _jacc(a["set"], b["set"])
        tok[i, 1] = _jacc(x["set"], y["set"])
        tok[i, 2] = _jacc(x["nums"], y["nums"])
        tok[i, 3] = 1.0 if (x["nums"] and y["nums"] and not (x["nums"] & y["nums"])) else 0.0
        tok[i, 4] = 1.0 if (x["postal"] & y["postal"]) else 0.0
        tok[i, 5] = 1.0 if (x["postal"] and y["postal"] and not (x["postal"] & y["postal"])) else 0.0
        if ext:
            tok[i, 6] = (float("nan") if x["first"] is None or y["first"] is None
                         else float(x["first"] == y["first"]))
            tok[i, 7] = _near_num(x["numlist"], y["numlist"])
            tok[i, 8] = min(a["n"], b["n"]) / max(a["n"], b["n"], 1)
            core_q.append(a["core"]); core_c.append(b["core"])
            cc_q.append(a["concat"]); cc_c.append(b["concat"])
    X[:, 16:22] = tok[:, 0:6]
    X[:, 22] = np.fromiter((1.0 if a == b else 0.0 for a, b in zip(q_country, c_country)), np.float64, n)
    X[:, 23] = np.fromiter((1.0 if str(c).startswith("S3-") else 0.0 for c in c_id), np.float64, n)
    X[:, 24] = cols["support"]
    X[:, 25] = cols["priority"]
    X[:, 26] = cols["rank"]
    X[:, 27] = cols["cand_count"]
    X[:, 28] = cols["best_priority"] - cols["priority"]
    if ext:
        X[:, 29] = fz(core_q, core_c, fuzz.token_set_ratio)
        X[:, 30] = fz(cc_q, cc_c, fuzz.ratio)
        X[:, 31] = fz(cc_q, cc_c, fuzz.partial_ratio)
        X[:, 32] = cols["jw_name"]
        X[:, 33] = cols["jw_addr"]
        X[:, 34] = cols["ev_name"]
        X[:, 35] = cols["ev_addr"]
        X[:, 36] = cols["n_keys"]
        X[:, 37] = cols["evidence"]
        X[:, 38] = tok[:, 6]
        X[:, 39] = tok[:, 7]
        X[:, 40] = cols["c_name_indic"]
        X[:, 41] = cols["c_addr_indic"]
        X[:, 42] = cols["priority"] - cols["next_priority"]
        X[:, 43] = cols["best_priority"] - cols["second_priority"]
        X[:, 44] = cols["source_rank"]
        X[:, 45] = tok[:, 8]
    return X


def features_scalar(qn, qa, qc, cn, ca, cc, cid, row: dict, feature_set="v2", norm_mode="canon"):
    """Straightforward per-pair reference implementation (used only by the
    scalar/batched equivalence test)."""
    from rapidfuzz import fuzz

    f = []
    for a, b in ((qn, cn), (qa, ca)):
        f += [fuzz.ratio(a, b) / 100, fuzz.WRatio(a, b) / 100, fuzz.token_sort_ratio(a, b) / 100,
              fuzz.token_set_ratio(a, b) / 100, float(a != "" and a == b), float(not a), float(not b),
              min(len(a), len(b)) / max(len(a), len(b), 1)]
    nt1, nt2 = set(qn.split()), set(cn.split())
    at1, at2 = set(qa.split()), set(ca.split())
    nums1 = {t for t in qa.split() if t.isascii() and t.isdigit()}
    nums2 = {t for t in ca.split() if t.isascii() and t.isdigit()}
    pc1 = {x for x in nums1 if len(x) in (5, 6)}
    pc2 = {x for x in nums2 if len(x) in (5, 6)}

    def jac(x, y):
        return len(x & y) / len(x | y) if (x | y) else 0.0

    f += [jac(nt1, nt2), jac(at1, at2), jac(nums1, nums2),
          float(bool(nums1 and nums2) and not nums1 & nums2),
          float(bool(pc1 & pc2)), float(bool(pc1 and pc2) and not pc1 & pc2),
          float(qc == cc), float(str(cid).startswith("S3-")),
          row["support"], row["priority"], row["rank"], row["cand_count"],
          row["best_priority"] - row["priority"]]
    if feature_set == "v2":
        c1, c2 = name_core_tokens(qn, norm_mode), name_core_tokens(cn, norm_mode)
        l1 = [t for t in qa.split() if t.isascii() and t.isdigit()]
        l2 = [t for t in ca.split() if t.isascii() and t.isdigit()]
        f += [fuzz.token_set_ratio(" ".join(c1), " ".join(c2)) / 100,
              fuzz.ratio("".join(c1), "".join(c2)) / 100,
              fuzz.partial_ratio("".join(c1), "".join(c2)) / 100,
              row["jw_name"], row["jw_addr"], row["ev_name"], row["ev_addr"], row["n_keys"],
              row["evidence"],
              float("nan") if not l1 or not l2 else float(l1[0] == l2[0]),
              _near_num(l1, l2),
              float(row["c_name_indic"]), float(row["c_addr_indic"]),
              row["priority"] - row["next_priority"], row["best_priority"] - row["second_priority"],
              row["source_rank"],
              min(len(qn.split()), len(cn.split())) / max(len(qn.split()), len(cn.split()), 1)]
    return np.asarray(f, dtype=np.float32)


RETRIEVAL_NUMERIC = ["support", "priority", "rank", "cand_count", "best_priority", "jw_name", "jw_addr",
                     "ev_name", "ev_addr", "n_keys", "evidence", "c_name_indic", "c_addr_indic",
                     "next_priority", "second_priority", "source_rank"]


def _feature_chunk(args):
    """Pool worker: feature_matrix on a row slice (single-threaded RapidFuzz inside the worker)."""
    qn, qa, qc, cn, ca, cc, cid, cols, feature_set, norm_mode = args
    return feature_matrix(qn, qa, qc, cn, ca, cc, cid, cols, feature_set=feature_set, norm_mode=norm_mode, workers=1)


def features_for_candidates(cand, q_tbl, feature_set, norm_mode, workers=-1, pool=None, min_rows_per_chunk=20000):
    """Join retrieval output back to query strings and build the feature matrix.

    With a process pool the rows are split into contiguous chunks, each processed by the same
    feature_matrix function, and concatenated in order (identical output, ~#workers faster)."""
    qid = cand.column("qid").to_numpy()
    base = q_tbl.column("qid")[0].as_py() if q_tbl.num_rows else 0
    idx = qid - base
    qn = np.asarray(q_tbl.column("name").to_pylist(), dtype=object)[idx]
    qa = np.asarray(q_tbl.column("addr").to_pylist(), dtype=object)[idx]
    qc = np.asarray(q_tbl.column("country").to_pylist(), dtype=object)[idx]
    cols = {k: cand.column(k).to_numpy(zero_copy_only=False).astype(np.float64) for k in RETRIEVAL_NUMERIC}
    n = len(qid)
    if pool is not None and n >= 2 * min_rows_per_chunk:
        n_chunks = max(2, min(getattr(pool, "_processes", 8) * 2, n // min_rows_per_chunk))
        bounds = np.linspace(0, n, n_chunks + 1).astype(int)
        cn, ca = cand.column("c_name").to_pylist(), cand.column("c_addr").to_pylist()
        cc, cid = cand.column("c_country").to_pylist(), cand.column("cand_id").to_pylist()
        jobs = [(qn[a:b], qa[a:b], qc[a:b], cn[a:b], ca[a:b], cc[a:b], cid[a:b], {k: v[a:b] for k, v in cols.items()},
                 feature_set, norm_mode) for a, b in zip(bounds[:-1], bounds[1:])]
        return np.concatenate(pool.map(_feature_chunk, jobs))
    return feature_matrix(qn, qa, qc, cand.column("c_name").to_pylist(), cand.column("c_addr").to_pylist(),
                          cand.column("c_country").to_pylist(), cand.column("cand_id").to_pylist(), cols,
                          feature_set=feature_set, norm_mode=norm_mode, workers=workers)


# ---------------------------------------------------------------------------
# 6. Metric
# ---------------------------------------------------------------------------

def entity_scores(entity_idx, y, pred, n_true):
    """Per-entity F0.5 = 5TP/(5TP+4FP+FN); singletons: 1 if nothing predicted else 0.

    entity_idx: int array (pair -> entity index); y: bool truth per pair;
    pred: bool prediction per pair; n_true: #true links per entity (including
    true links that never reached the candidate set).
    """
    E = len(n_true)
    tp = np.bincount(entity_idx, weights=(pred & y), minlength=E)
    fp = np.bincount(entity_idx, weights=(pred & ~y), minlength=E)
    fn = n_true - tp
    denom = 5 * tp + 4 * fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(n_true > 0, np.where(denom > 0, 5 * tp / np.maximum(denom, 1e-12), 0.0),
                     np.where(tp + fp == 0, 1.0, 0.0))
    return s, tp, fp, fn


def summarize(entity_idx, y, pred, n_true):
    s, tp, fp, fn = entity_scores(entity_idx, y, pred, n_true)
    TP, FP, FN = int(tp.sum()), int(fp.sum()), int(fn.sum())
    return {"macro_f05": float(s.mean()), "pair_precision": TP / max(TP + FP, 1),
            "pair_recall": TP / max(TP + FN, 1), "TP": TP, "FP": FP, "FN": FN, "entities": int(len(n_true))}


def bootstrap_ci(scores, n_boot=2000, seed=2026, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(scores)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = scores[rng.integers(0, n, n)].mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def best_for_target_flags(qids, cand_ids, probs):
    """True where this query has the highest probability for the target among all
    queries that retrieved it (ties -> smallest qid). Exploits the dataset fact
    that every Source-2/3 record belongs to at most one Source-1 entity."""
    import pandas as pd
    df = pd.DataFrame({"q": np.asarray(qids), "t": np.asarray(cand_ids), "p": np.asarray(probs)})
    df["o"] = np.arange(len(df))
    df = df.sort_values(["t", "p", "q"], ascending=[True, False, True], kind="mergesort")
    first = ~df["t"].duplicated()
    flags = np.zeros(len(df), dtype=bool)
    flags[df["o"].to_numpy()[first.to_numpy()]] = True
    return flags


# ---------------------------------------------------------------------------
# 7. Output validation
# ---------------------------------------------------------------------------

def validate_outputs(matching_path, candidate_path, s1_ids_in_order):
    """Structural validation. Raises AssertionError on the first category of failure."""
    errors = []
    expected = list(s1_ids_in_order)
    cand_map = {}
    for path, header, is_cand in ((candidate_path, "source1_entity_id\tcandidate_entity_ids", True),
                                  (matching_path, "source1_entity_id\tmatched_entity_ids", False)):
        with open(path, "r", encoding="utf-8", newline="") as f:
            first = f.readline().rstrip("\n").rstrip("\r")
            if first != header:
                errors.append(f"{Path(path).name}: bad header {first!r}")
                continue
            order = []
            for ln, line in enumerate(f, start=2):
                line = line.rstrip("\n")
                if "\r" in line:
                    errors.append(f"{Path(path).name}:{ln}: carriage return"); break
                parts = line.split("\t")
                if len(parts) != 2:
                    errors.append(f"{Path(path).name}:{ln}: expected 2 columns, got {len(parts)}"); break
                s1, lst = parts
                order.append(s1)
                ids = lst.split(",") if lst else []
                if any((" " in x) or (x == "") for x in ids):
                    errors.append(f"{Path(path).name}:{ln}: empty id or space in list"); break
                if len(ids) != len(set(ids)):
                    errors.append(f"{Path(path).name}:{ln}: duplicate id in list"); break
                if any(not (x.startswith("S2-") or x.startswith("S3-")) for x in ids):
                    errors.append(f"{Path(path).name}:{ln}: id without S2-/S3- prefix"); break
                if is_cand:
                    cand_map[s1] = set(ids)
                elif not set(ids) <= cand_map.get(s1, set()):
                    errors.append(f"{Path(path).name}:{ln}: match not in candidate set for {s1}"); break
            if len(order) != len(expected):
                errors.append(f"{Path(path).name}: {len(order)} rows, expected {len(expected)}")
            elif order != expected:
                bad = next(i for i, (a, b) in enumerate(zip(order, expected)) if a != b)
                errors.append(f"{Path(path).name}: order differs from test input at data row {bad + 1}")
    if errors:
        raise AssertionError("OUTPUT VALIDATION FAILED:\n  " + "\n  ".join(errors))
    return True


# ---------------------------------------------------------------------------
# 8. Orchestration: splits, batched query processing, model, inference
# ---------------------------------------------------------------------------

def select_splits(s1_ids, audit_bp, calib_bp, train_bp, salt="split:"):
    """Deterministic entity-level split by stable hash bucket (per 10,000).

    buckets [0, audit_bp) -> audit, [audit_bp, audit_bp+calib_bp) -> calibration,
    next train_bp buckets -> model training. Disjoint by construction.
    """
    b = np.fromiter((digest_id(salt + x) % 10000 for x in s1_ids), dtype=np.int64, count=len(s1_ids))
    a0, c0, t0 = audit_bp, audit_bp + calib_bp, audit_bp + calib_bp + train_bp
    return {"audit": np.flatnonzero(b < a0), "calib": np.flatnonzero((b >= a0) & (b < c0)),
            "train": np.flatnonzero((b >= c0) & (b < t0))}


def inner_holdout_mask(s1_ids):
    """Grouped model holdout: digest_id('inner:' + qid) % 5 == 0 (about 20%)."""
    return np.fromiter((digest_id("inner:" + x) % 5 == 0 for x in s1_ids), dtype=bool, count=len(s1_ids))


def take_raw(raw, idx):
    import pyarrow as pa
    return raw.take(pa.array(np.asarray(idx, dtype=np.int64)))


def process_queries(con, raw, cfg, feature_set, pool=None, batch_size=20000, truth=None, model=None,
                    timer=None, log=print, workers=-1, keep_rids_table=None, qid_offset=0):
    """Normalise -> retrieve -> features (-> labels) (-> probabilities) for raw S1 rows.

    Returns dict: entity_ids (per query), qid (per pair, = qid_offset + row index in raw),
    cand_id, rank, X (float32), y (bool, if truth), p (float32, if model), n_true.
    """
    timer = timer or Timer()
    n = raw.num_rows
    ents = [to_text(e).strip() for e in raw.column("entity_id").to_pylist()]
    out = {"entity_ids": ents, "qid": [], "cand_id": [], "rank": [], "X": [], "p": [], "y": []}
    t_start = time.perf_counter()
    for b0 in range(0, n, batch_size):
        sub = raw.slice(b0, batch_size)
        with timer("normalization"):
            nt = normalize_records(sub.column(0).to_pylist(), sub.column(1).to_pylist(), sub.column(2).to_pylist(),
                                   sub.column(3).to_pylist(), cfg["norm_mode"], pool=pool, chunk=2500)
            q = make_query_table(nt, qid_offset + b0)
        with timer("candidate_generation"):
            cand = retrieve(con, q, cfg, keep_rids_table=keep_rids_table)
        with timer("feature_generation"):
            X = features_for_candidates(cand, q, feature_set, cfg["norm_mode"], workers=workers, pool=pool)
        qid = cand.column("qid").to_numpy().astype(np.int64)
        cid = cand.column("cand_id").to_pylist()
        out["qid"].append(qid)
        out["cand_id"].extend(cid)
        out["rank"].append(cand.column("rank").to_numpy().astype(np.int16))
        out["X"].append(X)
        if truth is not None:
            out["y"].append(np.fromiter((c in truth.get(ents[k - qid_offset], ()) for k, c in zip(qid, cid)),
                                        dtype=bool, count=len(cid)))
        if model is not None:
            with timer("prediction"):
                out["p"].append(model.predict_proba(X)[:, 1].astype(np.float32))
        done = min(b0 + batch_size, n)
        el = time.perf_counter() - t_start
        log(f"  queries {done:,}/{n:,}  pairs {sum(len(x) for x in out['qid']):,}  {el:.0f}s  "
            f"({done / max(el, 1e-9):.0f} q/s)")
    nf = len(feature_names(feature_set))
    out["qid"] = np.concatenate(out["qid"]) if out["qid"] else np.zeros(0, np.int64)
    out["rank"] = np.concatenate(out["rank"]) if out["rank"] else np.zeros(0, np.int16)
    out["X"] = np.concatenate(out["X"]) if out["X"] else np.zeros((0, nf), np.float32)
    out["y"] = np.concatenate(out["y"]) if out["y"] else None
    out["p"] = np.concatenate(out["p"]) if out["p"] else None
    if truth is not None:
        out["n_true"] = np.array([len(truth.get(e, ())) for e in ents], dtype=np.float64)
    out["timer"] = dict(timer.t)
    return out


def save_block(path, block):
    """Save a process_queries result to .npz (atomic)."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, qid=block["qid"], rank=block["rank"], X=block["X"],
             y=block["y"] if block.get("y") is not None else np.zeros(0, bool),
             n_true=block.get("n_true", np.zeros(0)), entity_ids=np.asarray(block["entity_ids"], dtype=object),
             cand_id=np.asarray(block["cand_id"], dtype=object))
    os.replace(tmp, path)


def load_block(path):
    z = np.load(path, allow_pickle=True)
    return {"qid": z["qid"], "rank": z["rank"], "X": z["X"], "y": z["y"] if z["y"].size else None,
            "n_true": z["n_true"], "entity_ids": list(z["entity_ids"]), "cand_id": list(z["cand_id"])}


def catboost_params(base: dict, device: str, gpu_devices="0", gpu_ram_part=0.90):
    p = dict(base)
    if device == "GPU":
        p.update(task_type="GPU", devices=str(gpu_devices), gpu_ram_part=float(gpu_ram_part))
    else:
        p.update(task_type="CPU", thread_count=-1)
    return p


def fit_catboost(X, y, base_params, device, X_eval=None, y_eval=None, gpu_devices="0", gpu_ram_part=0.90,
                 log=print):
    """Fit CatBoost; on GPU failure retry with lower gpu_ram_part, then fall back to CPU.
    Returns (model, device_used). Never returns a partially trained model."""
    from catboost import CatBoostClassifier, CatBoostError, Pool

    pos = int(y.sum())
    if pos == 0 or pos == len(y):
        raise ValueError(f"training labels have a single class (positives={pos}, rows={len(y)})")
    train_pool = Pool(X, y.astype(np.int32))
    eval_pool = Pool(X_eval, y_eval.astype(np.int32)) if X_eval is not None else None
    attempts = []
    if device == "GPU":
        attempts += [("GPU", gpu_ram_part), ("GPU", max(0.5, gpu_ram_part - 0.25))]
    attempts.append(("CPU", None))
    last = None
    for dev, part in attempts:
        model = None
        try:
            params = catboost_params(base_params, dev, gpu_devices, part or gpu_ram_part)
            if eval_pool is None:
                params.pop("early_stopping_rounds", None)
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None, verbose=False)
            if dev != device:
                log(f"WARNING: requested {device} but trained on {dev}")
            return model, dev
        except (CatBoostError, RuntimeError, MemoryError) as e:
            last = e
            log(f"WARNING: CatBoost {dev} training failed ({type(e).__name__}: {str(e)[:200]}); trying next option")
            del model
            gc.collect()
    raise RuntimeError(f"CatBoost training failed on every device: {last!r}")


def benchmark_devices(X, y, base_params, iterations=200, rows=300_000, gpu_devices="0", gpu_ram_part=0.90,
                      seed=2026, log=print):
    """Time CatBoost CPU vs GPU on a representative subset. Returns (choice, timings)."""
    from catboost import CatBoostClassifier, Pool
    from catboost.utils import get_gpu_device_count

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y), size=min(rows, len(y)), replace=False))
    pool = Pool(X[idx], y[idx].astype(np.int32))
    timings = {}
    try:
        n_gpu = get_gpu_device_count()
    except Exception:
        n_gpu = 0
    for dev in ["CPU"] + (["GPU"] if n_gpu > 0 else []):
        params = catboost_params({**base_params, "iterations": iterations}, dev, gpu_devices, gpu_ram_part)
        params.pop("early_stopping_rounds", None)
        try:
            t0 = time.perf_counter()
            CatBoostClassifier(**params).fit(pool, verbose=False)
            timings[dev] = time.perf_counter() - t0
        except Exception as e:  # CUDA errors, driver problems ...
            timings[dev] = float("inf")
            log(f"WARNING: {dev} benchmark failed: {type(e).__name__}: {str(e)[:200]}")
        log(f"  {dev}: {timings[dev]:.2f}s for {iterations} iterations on {len(idx):,} rows")
    choice = min(timings, key=timings.get)
    if not math.isfinite(timings[choice]):
        raise RuntimeError("CatBoost failed on every device during benchmark")
    return choice, timings


def calibrate_threshold(entity_idx, y, p, n_true, extra_mask=None, grid=None):
    """Grid-search the probability threshold maximising macro-F0.5 (ties -> higher threshold)."""
    grid = np.round(np.arange(0.05, 0.9901, 0.01), 2) if grid is None else grid
    best_t, best_s, curve = None, -1.0, []
    for t in grid:
        pred = p >= t
        if extra_mask is not None:
            pred = pred & extra_mask
        s = float(entity_scores(entity_idx, y, pred, n_true)[0].mean())
        curve.append((float(t), s))
        if s >= best_s - 1e-12:  # '>=' prefers the higher threshold on ties
            best_t, best_s = float(t), s
    return best_t, best_s, curve


def inference_run_signature(model_path, threshold, cfg, feature_set, batch_size, test_paths, uniqueness):
    return {"pipeline": PIPELINE_VERSION, "model_sha256": file_sha256(model_path), "threshold": threshold,
            "cfg": cfg, "feature_set": feature_set, "batch_size": batch_size,
            "inputs": [file_fingerprint(p) for p in test_paths], "target_uniqueness": uniqueness}


def run_inference(con, raw, cfg, feature_set, model, parts_dir, run_sig: dict, pool=None, batch_size=20000,
                  timer=None, log=print, workers=-1):
    """Resumable batched inference. Each finished batch writes NNNNNN.pairs.parquet (every scored
    candidate pair) and NNNNNN.candidates.tsv, then the NNNNNN.done marker. Final matches are made
    by assemble_outputs because the one-owner-per-target rule needs all batches."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    timer = timer or Timer()
    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    sig_path = parts_dir / "run_signature.json"
    sig_s = json.dumps(run_sig, sort_keys=True, indent=1, default=str)
    if sig_path.exists():
        if sig_path.read_text(encoding="utf-8") != sig_s:
            raise RuntimeError(f"{parts_dir} holds parts from a different model/threshold/config. "
                               f"Change RUN_TAG or delete that directory deliberately.")
    else:
        if any(parts_dir.glob("*.done")):
            raise RuntimeError(f"{parts_dir} has .done parts but no signature; refusing to mix.")
        sig_path.write_text(sig_s, encoding="utf-8")
    n = raw.num_rows
    nb = (n + batch_size - 1) // batch_size
    t0 = time.perf_counter()
    skipped = 0
    for bi in range(nb):
        done = parts_dir / f"{bi:06d}.done"
        if done.exists():
            skipped += 1
            continue
        b0 = bi * batch_size
        sub = raw.slice(b0, batch_size)
        res = process_queries(con, sub, cfg, feature_set, pool=pool, batch_size=batch_size, model=model,
                              timer=timer, log=lambda *a: None, workers=workers, qid_offset=b0)
        with timer("file_writing"):
            tbl = pa.table({"qid": pa.array(res["qid"].astype(np.int32)), "cand_id": pa.array(res["cand_id"]),
                            "rank": pa.array(res["rank"]), "p": pa.array(res["p"])})
            tmp = parts_dir / f"{bi:06d}.pairs.parquet.tmp"
            pq.write_table(tbl, str(tmp))
            os.replace(tmp, parts_dir / f"{bi:06d}.pairs.parquet")
            lists = {}
            for q, c in zip(res["qid"].tolist(), res["cand_id"]):
                lists.setdefault(q, []).append(c)
            tmp = parts_dir / f"{bi:06d}.candidates.tsv.tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                for k, e in enumerate(res["entity_ids"]):
                    f.write(f"{e}\t{','.join(lists.get(b0 + k, []))}\n")
            os.replace(tmp, parts_dir / f"{bi:06d}.candidates.tsv")
            done.write_text("ok", encoding="utf-8")
        el = time.perf_counter() - t0
        n_new = bi + 1 - skipped
        eta = el / n_new * (nb - bi - 1)
        log(f"  batch {bi + 1}/{nb} done ({min(b0 + batch_size, n):,} queries; {el / 60:.1f} min this session; "
            f"ETA {eta / 60:.1f} min)")
    if skipped:
        log(f"  resumed: reused {skipped} completed batch(es)")
    return timer


def assemble_outputs(parts_dir, s1_ids, threshold, uniqueness, matching_path, candidate_path, threads=8):
    """Combine batch parts -> matching_results.tsv / candidate_pairs.tsv in test-input order."""
    import duckdb

    parts_dir = Path(parts_dir)
    n = len(s1_ids)
    files = sorted(parts_dir.glob("*.pairs.parquet"))
    dones = sorted(parts_dir.glob("*.done"))
    if not files or len(files) != len(dones):
        raise RuntimeError(f"{len(files)} part files but {len(dones)} .done markers in {parts_dir}")
    con = duckdb.connect()
    con.execute(f"SET threads={int(threads)}")
    glob = str(parts_dir / "*.pairs.parquet").replace("\\", "/")
    best = "row_number() OVER (PARTITION BY cand_id ORDER BY p DESC, qid) = 1" if uniqueness else "TRUE"
    con.execute(f"CREATE TABLE pairs AS SELECT qid, cand_id, rank, p, {best} AS is_best FROM read_parquet('{glob}')")
    n_pairs, max_qid, dup = con.execute(
        "SELECT count(*), max(qid), count(*) - count(DISTINCT (qid, cand_id)) FROM pairs").fetchone()
    if max_qid is not None and max_qid >= n:
        raise RuntimeError(f"part files reference qid {max_qid} >= number of test rows {n}")
    if dup:
        raise RuntimeError(f"{dup} duplicated (qid, cand_id) pairs across parts")
    cand_map = dict(con.execute("SELECT qid, string_agg(cand_id, ',' ORDER BY rank) FROM pairs GROUP BY qid").fetchall())
    match_map = dict(con.execute(
        f"SELECT qid, string_agg(cand_id, ',' ORDER BY rank) FROM pairs WHERE p >= {float(threshold)!r} AND is_best "
        f"GROUP BY qid").fetchall())
    con.close()
    for path, header, mp_ in ((candidate_path, "source1_entity_id\tcandidate_entity_ids", cand_map),
                              (matching_path, "source1_entity_id\tmatched_entity_ids", match_map)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(header + "\n")
            for k, e in enumerate(s1_ids):
                f.write(f"{e}\t{mp_.get(k, '')}\n")
        os.replace(tmp, path)
    return {"pairs": int(n_pairs), "entities_with_candidates": len(cand_map),
            "entities_with_matches": len(match_map),
            "predicted_links": sum(v.count(",") + 1 for v in match_map.values())}
