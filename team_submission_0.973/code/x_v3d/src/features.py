"""Pair features for the matcher (v3).

v1/v2 features are kept (now on v3-normalized text, zero-stripped numbers). v3 adds evidence the
audit found missing (work/audit/DIAGNOSIS.md, H5/H7):

  word evidence    the generator builds negative "siblings" by adding words such as holdings/group/
                   industries to a real name. For every name token the candidate ADDS (or drops)
                   relative to the S1 name, a log-odds  log P(token | false cand) - log P(token | true)
                   learned on fit-split S1 that the matcher never trains on (word_evidence.py).
  legal forms      legal-form tokens added / removed (siblings add them 4-12x more often)
  main number      relation of the main house numbers: equal / one digit dropped / off by 1-3 / other
  number prefix    candidate address starts with an added "H.NO 220", "No 35", "#72" style prefix
  glued names      similarity of names with spaces removed (domain forms: gomezmercator.com)
  street only      address similarity with all digits removed
"""
import math
import os
import pickle
import re

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein as Lev

from norm import ngrams, nums, raw_name_tokens, clean_name, strip_po
from shared_index import ROUTE_NAME, ROUTE_ADDR, ROUTE_NUM, ROUTE_NKEY, ROUTE_AKEY, ROUTE_NWORD

FEAT_NAMES = [
    "n_tokset", "n_toksort", "n_partial", "n_ratio", "n_3gram_jac", "n_lenratio",
    "n_first_tok_eq", "n_tfidf_cos", "n_acronym",
    "a_tokset", "a_toksort", "a_partial", "a_num_jac", "a_num_contain", "a_num_big",
    "x_script_eq", "n_exact_norm", "n_tokset_eq", "a_tok_contain",
    "a_tfidf_cos", "r_name", "r_addr", "r_num", "n_routes",
    "rank_name", "rank_addr", "n_cands", "n_close_name",
    "q_name_freq", "t_name_freq",
    "n_idf_jac", "n_idf_shared_max", "n_idf_unshared_max",
    "a_num_contra", "q_addr_empty", "t_addr_empty", "q_indic", "t_indic",
    # v3
    "ev_add_max", "ev_add_sum", "ev_rm_max", "n_added", "n_removed", "legal_add", "legal_rm",
    "mnum_rel", "mnum_logdiff", "t_prefix_num", "n_nospace_ratio", "a_street_tokset",
    "r_nkey", "r_akey", "r_nword", "nword_cos",
]

LEGAL_FORMS = {"private", "pvt", "limited", "ltd", "llp", "llc", "inc", "incorporated", "corp", "corporation",
               "company", "co", "pc", "pllc", "plc", "lp", "pa", "gmbh", "sa", "bv", "sarl", "sas", "sasu",
               "eurl", "sci", "snc", "ei", "cie"}
PREFIX_NUM = re.compile(r"^\s*(?:h\.?\s?no\.?|hn|door no\.?|plot|no\.?|n[°º]|#+)\s*[\w\-/]*\d", re.I)
DIGITS = re.compile(r"\d+")

EVIDENCE = {"add": {}, "rm": {}}


def load_evidence(path=None):
    path = path or os.environ.get("ER_WORD_EVIDENCE") or         os.path.join(os.path.dirname(os.path.abspath(__file__)), "word_evidence.pkl")
    if os.path.isfile(path):
        with open(path, "rb") as f:
            EVIDENCE.update(pickle.load(f))


load_evidence()


def main_number(raw_addr):
    """Longest digit group, zero-stripped (the house/plot number in almost all records)."""
    x = [d.lstrip("0") or "0" for d in DIGITS.findall(strip_po(raw_addr))]
    return max(x, key=len) if x else None


def token_diff(qtoks, ttoks):
    """Tokens only on one side, ignoring pairs within edit distance 1 (typos)."""
    qs, ts = set(qtoks), set(ttoks)
    add = [w for w in ts - qs if not any(len(w) >= 3 and Lev.distance(w, x) <= 1 for x in qs - ts)]
    rm = [w for w in qs - ts if not any(len(w) >= 3 and Lev.distance(w, x) <= 1 for x in ts - qs)]
    return add, rm


def mnum_relation(a, b):
    if a is None or b is None:
        return 0.0, 0.0
    if a == b:
        return 1.0, 0.0
    diff = math.log1p(abs(int(a[:12]) - int(b[:12])))
    if len(a) != len(b) and Lev.distance(a, b) == 1:
        return 2.0, diff
    if abs(int(a[:12]) - int(b[:12])) <= 3:
        return 3.0, diff
    return 4.0, diff


class QueryInfo:
    """Everything about one S1 record that every candidate pair reuses."""
    __slots__ = ("nn", "na", "grams", "toks", "idf", "u", "big", "atoks", "indic", "addr_empty",
                 "name_freq", "raw_toks", "mnum", "nospace", "street", "prefix")

    def __init__(self, q, nn, na, ci, indic):
        self.nn, self.na = nn, na
        self.grams = ngrams(nn)
        self.toks = set(nn.split())
        self.idf = {t: ci.idf(t) for t in self.toks}
        self.u = nums(q[2])
        self.big = {x for x in self.u if len(x) >= 3}
        self.atoks = set(na.split())
        self.indic = indic
        self.addr_empty = not q[2].strip()
        self.name_freq = ci.namefreq.get(nn)
        self.raw_toks = raw_name_tokens(q[1])
        self.mnum = main_number(q[2])
        self.nospace = clean_name(q[1]).replace(" ", "")
        self.street = DIGITS.sub(" ", na)
        self.prefix = bool(PREFIX_NUM.match(q[2]))


def _acronym(ta, tb):
    if len(ta) >= 2 and len(tb) == 1:
        return 1.0 if "".join(t[0] for t in ta) == tb[0] else 0.0
    if len(tb) >= 2 and len(ta) == 1:
        return 1.0 if "".join(t[0] for t in tb) == ta[0] else 0.0
    return 0.0


def featurize(qi, t, name_cos, addr_cos, bits, rank_n, rank_a, n_cands, n_close, nword_cos=0.0):
    """qi: QueryInfo; t: CountryIndex.record() tuple."""
    n1, n2, a1, a2 = qi.nn, t[3], qi.na, t[4]
    tl2 = n2.split()
    t2 = set(tl2)
    g2 = ngrams(n2)
    u2 = nums(t[2])
    b2 = {x for x in u2 if len(x) >= 3}
    at2 = set(a2.split())
    L = max(len(n1), len(n2)) or 1

    t_idf = dict(zip(tl2, t[7]))
    shared = qi.toks & t2
    w = lambda tok: qi.idf[tok] if tok in qi.idf else t_idf.get(tok, 0.0)
    w_shared = [w(x) for x in shared]
    w_other = [w(x) for x in (qi.toks ^ t2)]
    w_union = sum(w_shared) + sum(w_other)
    gu = qi.grams | g2
    u1 = qi.u

    # v3 evidence
    traw = raw_name_tokens(t[1])
    add, rm = token_diff(qi.raw_toks, traw)
    ev_add = [EVIDENCE["add"].get(x, 0.0) for x in add]
    ev_rm = [EVIDENCE["rm"].get(x, 0.0) for x in rm]
    rel, ldiff = mnum_relation(qi.mnum, main_number(t[2]))
    street2 = DIGITS.sub(" ", a2)

    return [
        fuzz.token_set_ratio(n1, n2) / 100, fuzz.token_sort_ratio(n1, n2) / 100,
        fuzz.partial_ratio(n1, n2) / 100, fuzz.ratio(n1, n2) / 100,
        len(qi.grams & g2) / len(gu) if gu else 0.0,
        min(len(n1), len(n2)) / L,
        1.0 if (n1 and tl2 and n1.split(" ", 1)[0] == tl2[0]) else 0.0,
        name_cos, _acronym(n1.split(), tl2),
        fuzz.token_set_ratio(a1, a2) / 100, fuzz.token_sort_ratio(a1, a2) / 100,
        fuzz.partial_ratio(a1, a2) / 100,
        len(u1 & u2) / len(u1 | u2) if (u1 or u2) else 0.0,
        len(u1 & u2) / min(len(u1), len(u2)) if (u1 and u2) else 0.0,
        1.0 if (qi.big & b2) else 0.0,
        1.0 if qi.indic == t[5] else 0.0,
        1.0 if (n1 and n1 == n2) else 0.0,
        1.0 if (qi.toks and qi.toks == t2) else 0.0,
        len(qi.atoks & at2) / min(len(qi.atoks), len(at2)) if (qi.atoks and at2) else 0.0,
        addr_cos,
        1.0 if bits & ROUTE_NAME else 0.0, 1.0 if bits & ROUTE_ADDR else 0.0,
        1.0 if bits & ROUTE_NUM else 0.0,
        float(bin(bits).count("1")),
        float(rank_n), float(rank_a), float(n_cands), float(n_close),
        math.log1p(qi.name_freq), math.log1p(t[6]),
        sum(w_shared) / w_union if w_union else 0.0,
        max(w_shared, default=0.0), max(w_other, default=0.0),
        1.0 if (u1 and u2 and not (u1 & u2)) else 0.0,
        1.0 if qi.addr_empty else 0.0, 1.0 if not t[2].strip() else 0.0,
        1.0 if qi.indic else 0.0, 1.0 if t[5] else 0.0,
        # v3
        max(ev_add, default=0.0), sum(ev_add), max(ev_rm, default=0.0),
        float(len(add)), float(len(rm)),
        float(sum(1 for x in add if x in LEGAL_FORMS)), float(sum(1 for x in rm if x in LEGAL_FORMS)),
        rel, ldiff,
        1.0 if (PREFIX_NUM.match(t[2]) and not qi.prefix) else 0.0,
        fuzz.ratio(qi.nospace, clean_name(t[1]).replace(" ", "")) / 100,
        fuzz.token_set_ratio(qi.street, street2) / 100,
        1.0 if bits & ROUTE_NKEY else 0.0, 1.0 if bits & ROUTE_AKEY else 0.0,
        1.0 if bits & ROUTE_NWORD else 0.0, nword_cos,
    ]
