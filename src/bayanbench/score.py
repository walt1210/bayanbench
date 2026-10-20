"""Automatic scoring of model answers against official PSGC data.

Every answer gets one of three outcomes:
    correct      matches the official answer
    abstained    the model said it did not know
    outdated     matches an official answer that was true before a recent change
    wrong        a confident answer that does not match (a hallucination)
"""
import re
from difflib import SequenceMatcher

from .normalize import is_abstention, normalize_name, strip_accents

CORRECT, ABSTAINED, WRONG, OUTDATED = "correct", "abstained", "wrong", "outdated"


def _province_key(name: str) -> str:
    """Province comparison keeps the word 'city' so 'Quezon City' never matches 'Quezon' province."""
    s = normalize_name(name, drop_unit_words=False)
    s = re.sub(r"\b(province of|province|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def score_choice(answer, gold_options: list[str], kind: str = "place",
                 outdated_options: list[str] | None = None) -> str:
    """Single-answer questions (province, region, city or municipality).

    gold_options lists every acceptable official answer, for example a region's
    full name, roman numeral, and common name.
    """
    if is_abstention(answer):
        return ABSTAINED
    if kind == "unit_type":
        return CORRECT if _unit_type(str(answer)) == _unit_type(gold_options[0]) else WRONG
    key = _province_key if kind == "province" else normalize_name
    if key(str(answer)) in {key(g) for g in gold_options}:
        return CORRECT
    if outdated_options and key(str(answer)) in {key(g) for g in outdated_options}:
        return OUTDATED
    return WRONG


def _unit_type(text: str) -> str | None:
    """'City' / 'component city' / 'Municipality' -> 'city' or 'municipality'."""
    t = strip_accents(text).lower().strip()
    if t in {"mun", "municipality"}:   # PSGC level code for municipalities
        return "municipality"
    has_city, has_mun = "city" in t, "municipal" in t or "town" in t
    if has_city and not has_mun:
        return "city"
    if has_mun and not has_city:
        return "municipality"
    return None  # ambiguous answers such as "city or municipality" are never correct


def score_count(answer, gold: int, tolerance: float = 0.10) -> dict:
    """Numeric questions such as the number of barangays."""
    if is_abstention(answer):
        return {"outcome": ABSTAINED, "exact": False, "within_tolerance": False}
    try:
        value = int(re.sub(r"[^\d]", "", str(answer)))
    except ValueError:
        return {"outcome": WRONG, "exact": False, "within_tolerance": False}
    within = abs(value - gold) <= max(1, round(gold * tolerance))
    return {"outcome": CORRECT if value == gold else WRONG,
            "exact": value == gold, "within_tolerance": within, "value": value}


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def match_barangay(name: str, official: list[str], threshold: float = 0.88) -> str | None:
    """Return the official barangay this name refers to, or None if it is invented.

    Exact match after normalization first; otherwise a close spelling match
    (for typos such as 'Dau' vs 'Dau ' or 'Sto Rosario' vs 'Santo Rosario').
    The threshold is strict on purpose: a wrong barangay with a similar name
    should count as invented, not as correct.
    """
    n = normalize_name(name)
    if not n:
        return None
    by_key = {normalize_name(o): o for o in official}
    if n in by_key:
        return by_key[n]
    best = max(official, key=lambda o: _similar(n, normalize_name(o)), default=None)
    if best is not None and _similar(n, normalize_name(best)) >= threshold:
        return best
    return None


GENERIC_BARANGAY = re.compile(
    r"^(?:(?:barangay|brgy|bgy|zone|purok|district|sitio)\s*[0-9ivx]+(?:\s?[a-z])?|poblacion\b.*|[0-9]+)$")


def is_generic_barangay(name: str) -> bool:
    """Placeholder-style names that exist in many towns: 'Brgy. 1', 'Barangay 2', 'Poblacion Proper'.

    Checked on the lowercased, accent-free form, not the normalized one, because
    normalization removes the word 'barangay'.
    """
    t = re.sub(r"[^a-z0-9 ]+", " ", strip_accents(str(name)).lower())
    t = re.sub(r"\s+", " ", re.sub(r"\(.*?\)", " ", t)).strip()
    return bool(GENERIC_BARANGAY.match(t))


def score_barangays(answers, official: list[str]) -> dict:
    """'Name three barangays in X.' Each name is real (in X) or invented."""
    if is_abstention(answers):
        return {"outcome": ABSTAINED, "named": 0, "real": 0, "generic": 0, "invented": [], "matches": []}
    if isinstance(answers, str):
        answers = [a.strip() for a in re.split(r"[;,\n]", answers) if a.strip()]
    seen, real, invented, matches, generic = set(), 0, [], [], 0
    for a in answers:
        if is_abstention(a):
            continue
        if is_generic_barangay(a):
            generic += 1
        m = match_barangay(a, official)
        if m is None:
            invented.append(a)
        elif m not in seen:
            seen.add(m)
            real += 1
            matches.append((a, m))
    named = real + len(invented)
    outcome = CORRECT if named and not invented else (WRONG if invented else ABSTAINED)
    return {"outcome": outcome, "named": named, "real": real, "generic": generic,
            "invented": invented, "matches": matches}
