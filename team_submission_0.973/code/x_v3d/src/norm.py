"""Text normalization shared by blocking and matching (v3d).

v3d (measured on v3c's remaining errors, work/audit/DIAGNOSIS.md):
- Dotted acronyms are collapsed BEFORE punctuation removal. v3 removed dots first, so "S.A.R.L." became
  the four junk tokens "s a r l" and the legal-form patterns never matched; 5.4% of France and 4.3% of
  US S2/S3 names carry dotted forms (S.A.S, S.A.R.L., E.U.R.L., L.L.C., P.C., M.D.).
- Digits inside alphabetic name tokens are folded to look-alike letters (0->o 1->l 5->s 8->b): OCR swaps.
- PO box / PMB / BP numbers are removed from addresses (and from number features); "n" from "N°" is dropped.
- Address components pass through a learned, per-country component alias table (learn_components.py): multi-word
  state/region names and codes, place <-> parent municipality, France department <-> region.

v3 changes, each measured in the audit (work/audit/DIAGNOSIS.md):
- Generator sibling words (group, holdings, industries, enterprises, solutions, ...) are no longer
  deleted as legal suffixes. They appear on 0 of 64,600 true links but on 12% of orphan candidates;
  v2 erased the evidence. Only legal FORMS are stripped now (incl. French SAS/SASU/EURL/SCI/SNC).
- Numbers lose leading zeros ("003245" -> "3245"): 4.7% of true links are zero-padded and 45% of
  them were rejected by v2 because the number features called them contradictions.
- Address noise tokens are dropped: NULL / N/A / <NULL>, the "CITY" suffix, "no"/"h.no"/"door"
  prefixes, "bis"/"ter"; ordinal suffixes are removed (47th / 47nd -> 47) and spelled ordinals become
  digits. French street types are mapped to one form (r/rue, av/avenue, bd, all/allee, imp, rte, ...).
- Names: leading honorifics, "(ID: 123)" tags, "#"/"@" handles, glued "...Com" domains and
  consecutive duplicate tokens are removed; for "X trading as Y" style aliases the part after the
  marker is used.
- Only Latin combining accents are stripped (Indic vowel signs are kept, as in v2); tokens are
  canonicalized through alias tables learned from fit-split links (learn_aliases.py).
"""
import os
import pickle
import re
import unicodedata

LEGAL = r"""private|pvt|limited|ltd|llp|llc|l\.l\.c|l\.l\.p|inc|incorporated|corp|corporation|
company|co|pc|p\.c|pllc|plc|lp|l\.p|pa|p\.a|and\s+sons|gmbh|sa|s\.a|bv|sarl|s\.a\.r\.l|sas|s\.a\.s|
sasu|s\.a\.s\.u|eurl|e\.u\.r\.l|sci|s\.c\.i|snc|s\.n\.c|ei|cie|services|service|societe|et\s+fils|and\s+co"""
LEGAL_RE = re.compile(r"\b(?:" + LEGAL.replace("\n", "") + r")\b\.?", re.I)
TLD_RE = re.compile(r"\.(com|net|org|co|in|io|biz|info|us|fr)\b", re.I)
GLUED_COM = re.compile(r"^([a-z]{6,})com$")
DBA_RE = re.compile(r"^.*?\b(?:trading as|t/a|d/b/a|d\.b\.a\.?|dba:?|doing business as|a/k/a|aka|"
                    r"f/k/a|formerly known as)\s+", re.I)
IDTAG_RE = re.compile(r"\(\s*id:?\s*\d+\s*\)|\s-\s*\d{6,}\s*$", re.I)
HONOR_RE = re.compile(r"^\s*(?:shri|shree|smt|sri|dr|mr|mrs|ms|m/s|the)\b\.?\s+", re.I)
# \w misses Indic vowel signs and virama (categories Mn/Mc); keep the Indic blocks whole
PUNCT_RE = re.compile(r"[^\w\sऀ-෿]")
ACCENT_RE = re.compile(r"[̀-ͯ]")
WS_RE = re.compile(r"\s+")
ADDR_NOISE_RE = re.compile(r"<null>|\bnull\b|\bn/a\b|\b1/2\b", re.I)
ORD_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
NUM_RE = re.compile(r"\d+")
DOTTED_RE = re.compile(r"(?<![\w.])((?:[a-z]\.){2,}[a-z]?)\.?(?!\w)")
PO_RE = re.compile(r"\b(?:p\.?\s?o\.?\s?box|post box|pmb|bp|b\.p\.)\s*#?\s*\d+", re.I)
OCR_MAP = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b"})
ALPHA_RE = re.compile(r"[a-z]")
COMP_KEY_RE = re.compile(r"[^\w\sऀ-෿]")

SPELLED = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
           "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10"}
ADDR_ABBR = {
    "road": "rd", "street": "st", "saint": "st", "avenue": "ave", "av": "ave", "drive": "dr", "lane": "ln",
    "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "highway": "hwy", "floor": "flr", "building": "bldg",
    "block": "blk", "nagar": "ngr", "marg": "mrg", "phase": "ph", "sector": "sec", "apartment": "apt",
    "suite": "ste", "opposite": "opp", "rue": "r", "allee": "all", "impasse": "imp", "route": "rte",
    "place": "pl", "residence": "res", "chemin": "ch", "chem": "ch", "cours": "crs", "square": "sq",
    "quai": "q", "sainte": "ste",
    # dropped: prefixes and fillers that carry no identity
    "near": "", "no": "", "nº": "", "n°": "", "number": "", "house": "", "door": "", "hno": "", "city": "",
    "bis": "", "ter": "", "n": "",
}
ADDR_ABBR.update(SPELLED)

NAME_ALIAS, ADDR_ALIAS, COMP_ALIAS = {}, {}, {}


def load_aliases(path=None):
    """Load learned alias tables. Called at import; safe when the file is absent."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "aliases.pkl")
    NAME_ALIAS.clear()
    ADDR_ALIAS.clear()
    if os.path.isfile(path):
        with open(path, "rb") as f:
            tables = pickle.load(f)
        NAME_ALIAS.update(tables["name"])
        ADDR_ALIAS.update(tables["addr"])
    COMP_ALIAS.clear()
    cpath = os.environ.get("ER_COMP_ALIAS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "comp_alias.pkl")
    if os.path.isfile(cpath):
        with open(cpath, "rb") as f:
            COMP_ALIAS.update(pickle.load(f))


def _canon(tok, table):
    for _ in range(3):                 # follow short chains, never loop
        nxt = table.get(tok)
        if nxt is None or nxt == tok:
            break
        tok = nxt
    return tok


def script_of(s):
    """Dominant unicode script bucket of a string."""
    counts = {}
    for ch in s:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch).split()[0]
        except ValueError:
            continue
        b = name if name in ("LATIN", "DEVANAGARI", "BENGALI", "TELUGU", "TAMIL",
                             "GUJARATI", "KANNADA", "MALAYALAM", "ORIYA", "GURMUKHI",
                             "ARABIC", "CYRILLIC") else "OTHER"
        counts[b] = counts.get(b, 0) + 1
    return max(counts, key=counts.get) if counts else "NONE"


def base_norm(s):
    s = unicodedata.normalize("NFKD", str(s).replace("�", ""))
    s = ACCENT_RE.sub("", s)           # Latin accents only (ÍNDIA -> india)
    return unicodedata.normalize("NFC", s).casefold()


def undot(s):
    """'s.a.r.l.' -> 'sarl', 'l.l.c.' -> 'llc' (letters separated by dots; domains are untouched)."""
    return DOTTED_RE.sub(lambda m: m.group(1).replace(".", ""), s)


def comp_key(c):
    return WS_RE.sub(" ", COMP_KEY_RE.sub(" ", c)).strip()


def strip_po(s):
    return PO_RE.sub(" ", str(s))


def clean_name(s):
    """Raw name with generator wrappers removed (alias marker, ID tag, honorific, handle)."""
    s = undot(base_norm(s))
    s = DBA_RE.sub("", s)
    s = IDTAG_RE.sub(" ", s)
    s = s.strip().lstrip("#@").strip()
    s = HONOR_RE.sub("", s)
    return s


def _dedupe(toks):
    out = []
    for t in toks:
        if not out or out[-1] != t:
            out.append(t)
    return out


def raw_name_tokens(s):
    """Name tokens before alias mapping and legal-form removal (used for mining and evidence)."""
    toks = PUNCT_RE.sub(" ", TLD_RE.sub(" ", clean_name(s))).split()
    toks = [t.translate(OCR_MAP) if (ALPHA_RE.search(t) and not t.isdigit() and
                                     sum(ch.isalpha() for ch in t) >= 2) else t for t in toks]
    if len(toks) == 1:
        m = GLUED_COM.match(toks[0])
        if m:
            toks = [m.group(1)]
    return _dedupe(toks)


def raw_addr_tokens(s, country=None):
    s = ADDR_NOISE_RE.sub(" ", undot(strip_po(base_norm(s))))
    table = COMP_ALIAS.get(country) if country else None     # per country: "tn" is Tamil Nadu or Tennessee
    if table:
        s = ", ".join(table.get(comp_key(c), c) for c in s.split(","))
    s = ORD_RE.sub(r"\1", s)
    return PUNCT_RE.sub(" ", s).split()


def norm_name(s):
    toks = [_canon(t, NAME_ALIAS) for t in raw_name_tokens(s)]
    return WS_RE.sub(" ", LEGAL_RE.sub(" ", " ".join(toks))).strip()


def _num_tok(t):
    return (t.lstrip("0") or "0") if t.isdigit() else t


def norm_addr(s, country=None):
    toks = (ADDR_ABBR.get(t, t) for t in (_canon(t, ADDR_ALIAS) for t in raw_addr_tokens(s, country)))
    return " ".join(_num_tok(t) for t in toks if t)


def nums(s):
    """Digit groups without leading zeros ("003245" == "3245"); PO box / PMB numbers ignored."""
    return {x.lstrip("0") or "0" for x in NUM_RE.findall(strip_po(s))}


def ngrams(s, n=3):
    s = s.replace(" ", "")
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else ({s} if s else set())


load_aliases()
