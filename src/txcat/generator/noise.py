"""Templated bank-statement noise applied to a clean brand name. Patterns mirror those observed in
DC/Oklahoma descriptors: processor prefixes, alphanumeric reference codes, store numbers, city/state
suffixes, hard truncation at 22/25 chars, whitespace padding, vowel-dropped abbreviation."""

from __future__ import annotations

import re

import numpy as np

PREFIXES = [
    "SQ *",
    "TST* ",
    "PP*",
    "PAYPAL *",
    "PYPL*",
    "IN *",
    "DD *",
    "CLV*",
    "POS PURCHASE ",
    "",
]
PREFIX_P = [0.08, 0.06, 0.04, 0.04, 0.02, 0.03, 0.03, 0.02, 0.03, 0.65]
ALNUM = np.array(list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"))


def _ref_code(rng: np.random.Generator) -> str:
    n = int(rng.integers(5, 10))
    return "".join(rng.choice(ALNUM, size=n))


def _abbrev(name: str, rng: np.random.Generator) -> str:
    words = name.split()
    out = []
    for w in words:
        if len(w) > 5 and rng.random() < 0.5:
            w = w[0] + re.sub(r"[AEIOU]", "", w[1:].upper())
        out.append(w.upper())
    return " ".join(out)


def noisy_descriptor(name: str, city: str, state: str, rng: np.random.Generator) -> str:
    s = name.upper()
    if rng.random() < 0.15:
        s = _abbrev(name, rng)
    s = np.random.default_rng(int(rng.integers(0, 2**31))).choice(PREFIXES, p=PREFIX_P) + s
    r = rng.random()
    if r < 0.30:
        s += f" #{int(rng.integers(1, 9999))}"
    elif r < 0.55:
        s += f" {_ref_code(rng)}"
    elif r < 0.70:
        s += f" {int(rng.integers(100, 99999)):05d}"
    r = rng.random()
    if r < 0.35:
        s += f" {city.upper()} {state}"
    elif r < 0.50:
        s += f" {state}"
    if rng.random() < 0.20:
        s = s[: int(rng.choice([22, 25]))]
    if rng.random() < 0.10:
        parts = s.split(" ", 1)
        s = parts[0] + " " * int(rng.integers(3, 9)) + (parts[1] if len(parts) > 1 else "")
    return s.strip()[:40]
