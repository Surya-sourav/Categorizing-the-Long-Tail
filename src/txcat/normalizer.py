"""Rule-based merchant descriptor normalization. Conservative: strips processor noise and
reference codes, never maps brands. Every applied rule is recorded in ``steps`` for audit."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Processor / channel prefixes. Matched at string start, case-insensitive, optional space/asterisk.
PREFIXES = [
    "SQ *", "SQ*", "TST* ", "TST*", "PAYPAL *", "PAYPAL*", "PYPL*", "PP*", "DD *", "DD*",
    "IN *", "IN*", "CLV*", "IC*", "POS PURCHASE ", "POS ", "DBT PURCHASE ",
    "CKCD ", "PURCHASE ", "DEBIT CARD PURCHASE ",
]
_PREFIX_RE = re.compile(r"^(?:" + "|".join(re.escape(p) for p in PREFIXES) + r")", re.IGNORECASE)
_STORE_NUM_RE = re.compile(r"#\s*\d+")
_TLD_RE = re.compile(r"\.(?:COM|NET|ORG|CO|US|IO)$", re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b")
_PAD_RE = re.compile(r"\bQQQ\b")
_WS_RE = re.compile(r"\s+")


@dataclass
class NormalizeResult:
    text: str
    steps: list[str] = field(default_factory=list)


def _is_ref_code(tok: str) -> bool:
    """Digit-heavy tokens are store/reference codes, not merchant words."""
    digits = sum(c.isdigit() for c in tok)
    letters = sum(c.isalpha() for c in tok)
    if tok.isdigit() and len(tok) >= 3:
        return True
    return digits >= 3 and letters >= 1 and "-" not in tok and "'" not in tok


def normalize_merchant(raw: str) -> NormalizeResult:
    """Normalize a raw card descriptor. Returns cleaned uppercase text plus the rules applied."""
    steps: list[str] = []
    s = raw.strip()
    if not s:
        return NormalizeResult("", steps)
    s = s.upper()
    steps.append("upper")

    m = _PREFIX_RE.match(s)
    if m:
        s = s[m.end():]
        steps.append(f"strip_prefix:{m.group(0).strip().upper()}")

    if _STORE_NUM_RE.search(s):
        s = _STORE_NUM_RE.sub(" ", s)
        steps.append("drop_store_number:#")
    if "*" in s:
        s = s.replace("*", " ")
        steps.append("replace_asterisk")
    toks_url = []
    for tok in s.split():
        if _TLD_RE.search(tok):
            if tok.count(".") >= 2 or tok.upper().startswith("WWW."):
                steps.append(f"drop_url:{tok}")
                continue  # subdomain-ish token (HELP.UBER.COM) carries no merchant identity
            tok = _TLD_RE.sub("", tok)  # AMAZON.COM -> AMAZON
            steps.append("strip_tld")
        toks_url.append(tok)
    s = " ".join(toks_url)
    if _DATE_RE.search(s):
        s = _DATE_RE.sub(" ", s)
        steps.append("drop_date")
    if _PAD_RE.search(s):
        s = _PAD_RE.sub(" ", s)
        steps.append("drop_padding:QQQ")

    toks = s.split()
    kept = [t for t in toks if not _is_ref_code(t)]
    if len(kept) != len(toks):
        steps.append(f"drop_ref_codes:{len(toks) - len(kept)}")
    s = " ".join(kept)

    s = s.strip(" *-/|,.")
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        # everything was noise; fall back to the uppercased raw minus punctuation
        s = _WS_RE.sub(" ", re.sub(r"[*#]", " ", raw.upper())).strip()
        steps.append("fallback_raw")
    return NormalizeResult(s, steps)
