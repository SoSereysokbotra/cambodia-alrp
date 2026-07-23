"""
src/recognition/plate_format.py
===============================
IMPROVEMENT_PLAN_V2 Phase 1 — the Cambodian plate-number grammar, in one place.

Why this exists
---------------
The CRNN decodes freely: any sequence of charset characters is a possible output.
But a real Cambodian plate number has a rigid shape, and measurement on this
project's own data showed the gap is large and exploitable:

    Ground truth (622 human labels)      Confident live reads (crnn_conf >= 0.70, n=5952)
    ---------------------------------    ------------------------------------------------
    DL-DDDD    345  (55.5%)              DL-DDDD    4847   legal
    D-DDDD     248  (39.9%)              DL-DDD      306   IMPOSSIBLE
    DLL-DDDD    20  ( 3.2%)              D-DDDD      302   legal
    vanity       6  ( 1.0%)              DL-DD        96   IMPOSSIBLE
    artefacts    3  ( 0.5%)              DLL-DDDD     70   legal
                                         D-DDD        36   IMPOSSIBLE
                                         DDDDD        34   IMPOSSIBLE
                                         DDDDDD       32   IMPOSSIBLE
                                         L-DDDD       29   IMPOSSIBLE
                                         DL-D-DD      23   IMPOSSIBLE

(D = digit, L = Latin letter.) Roughly 800 reads cleared the REC-005 confidence
gate in a format no Cambodian plate has. That population overlaps the benchmark's
dominant failure mode ("length-wrong 23 of 42"), and it is detectable with no
model change at all.

Keeping the table data-driven
-----------------------------
`PATTERNS` below carries the observed count for each rule. When a new plate series
appears, add a row — do not scatter regexes through the pipeline.

Safety contract (this is the point of the module)
-------------------------------------------------
`is_valid()` only ever REJECTS. Callers use it to scale a confidence DOWN, which
can only move a read toward REVIEW_REQUIRED, never toward ENTRY_ALLOWED. It
therefore cannot introduce a false accept.

`normalise()` INSERTS characters (a missing dash), so it can turn a non-matching
read into one that exact-matches a registered plate. That is a repair, not a
rejection, and it is deliberately NOT used by the level-1 validation path. Wire it
only behind `gate.format_repair`, and route repaired reads to REVIEW_REQUIRED the
same way ROADMAP 1.2 routes near-matches.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# The grammar. (name, compiled regex, observed count in the 622 real labels)
# --------------------------------------------------------------------------- #
#
# STANDARD — the province-series plates. These are the ones that carry a Khmer
# province line and are what the gate normally sees. 613/622 = 98.6% of labels.
STANDARD_PATTERNS: list[tuple[str, re.Pattern, int]] = [
    ("D L - D D D D",     re.compile(r"^\d[A-Z]-\d{4}$"),      345),
    ("D - D D D D",       re.compile(r"^\d-\d{4}$"),           248),
    ("D L L - D D D D",   re.compile(r"^\d[A-Z]{2}-\d{4}$"),    20),
]

# VANITY — the CAMBODIA / "Other" series (province class 25, no Khmer prefix)
# carries custom text: COVI19, HENGHENG, HYWAZA9, ELDC865, SELA GTR.
# Only 6 of 622, but they are legitimate plates: excluding them would push real
# cars to REVIEW for no benefit, and no observed *bad* read looks like this (every
# impossible pattern above is digit-led), so allowing them costs zero detection power.
VANITY_PATTERN = re.compile(r"^[A-Z0-9]{4,8}$")
VANITY_MIN_LETTERS = 3

# Known GROUND-TRUTH artefacts, listed so nobody "fixes" the grammar to admit them:
#   "6667" / "-7495"  -> the labelled plate was partly out of frame; the label is a
#                        fragment, not a plate format.
#   "10-1152"         -> single instance; likely a mislabelled "1D-1152".
# These are data problems, not grammar gaps.


def _clean(text: str) -> str:
    """Upper-case, drop spaces, collapse repeated dashes. Never invents characters."""
    if not text:
        return ""
    t = text.upper().replace(" ", "")
    t = re.sub(r"-{2,}", "-", t)
    return t.strip()


def is_valid(text: str, allow_vanity: bool = True) -> bool:
    """True if `text` is a shape a real Cambodian plate number can take.

    Pure predicate: never modifies the input, never raises.
    """
    t = _clean(text)
    if not t:
        return False
    for _name, rx, _n in STANDARD_PATTERNS:
        if rx.match(t):
            return True
    if allow_vanity and VANITY_PATTERN.match(t):
        if sum(c.isalpha() for c in t) >= VANITY_MIN_LETTERS:
            return True
    return False


def signature(text: str) -> str:
    """Pattern signature for logging/analysis: '3E-6306' -> 'DL-DDDD'."""
    t = _clean(text)
    return re.sub(r"[0-9]", "D", re.sub(r"[A-Z]", "L", t))


def matched_pattern(text: str) -> str | None:
    """Name of the STANDARD pattern `text` matches, else 'vanity', else None."""
    t = _clean(text)
    for name, rx, _n in STANDARD_PATTERNS:
        if rx.match(t):
            return name
    if VANITY_PATTERN.match(t) and sum(c.isalpha() for c in t) >= VANITY_MIN_LETTERS:
        return "vanity"
    return None


def normalise(text: str) -> str:
    """Best-effort canonical form — REPAIR, not validation. See the safety note
    in the module docstring before wiring this into a gate path.

    Only ever inserts the series dash at the position the grammar implies; it
    never invents an alphanumeric character. Returns the cleaned input unchanged
    when no single-dash insertion makes it legal.
    """
    t = _clean(text)
    if not t or is_valid(t):
        return t
    if "-" in t:
        return t                     # already has a dash; a second one won't help
    # Try inserting a dash at each position; accept only if the result is legal
    # AND unambiguous (exactly one position works) — an ambiguous repair is a
    # guess, and a guess must not reach the gate.
    hits = [t[:i] + "-" + t[i:] for i in range(1, len(t))
            if is_valid(t[:i] + "-" + t[i:], allow_vanity=False)]
    return hits[0] if len(hits) == 1 else t


if __name__ == "__main__":       # quick self-check: python src/recognition/plate_format.py
    legal = ["3E-6306", "2-4785", "1GE-3348", "COVI19", "HENGHENG", "HYWAZA9"]
    illegal = ["1M-777", "3E-66", "2-123", "12345", "123456", "L-6023", "1M-7-76", ""]
    for s in legal:
        assert is_valid(s), f"should be legal: {s!r}"
    for s in illegal:
        assert not is_valid(s), f"should be illegal: {s!r}"
    assert normalise("1M7776") == "1M-7776", normalise("1M7776")
    assert normalise("24785") == "2-4785", normalise("24785")
    assert normalise("1M-777") == "1M-777"          # unrepairable, unchanged
    assert signature("3E-6306") == "DL-DDDD"
    assert matched_pattern("2-4785") == "D - D D D D"
    print("plate_format self-check: OK")
