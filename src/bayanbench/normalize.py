"""Name normalization for Philippine place names.

Model answers and official PSGC names differ in harmless ways:
    "City of San Fernando" vs "San Fernando City" vs "San Fernando"
    "Dasmariñas" vs "Dasmarinas"
    "Sto. Tomas" vs "Santo Tomas"
    "Poblacion (Pob.)" vs "Poblacion"
The scorer should accept these as the same place, but never accept a
different place. Normalization therefore only removes formatting, never
meaning.
"""
import re
import unicodedata

ABBREVIATIONS = {
    "sto": "santo", "sta": "santa", "sn": "san", "gen": "general",
    "pres": "president", "pob": "poblacion", "brgy": "", "bgy": "",
    "barangay": "", "mt": "mount", "ft": "fort", "dr": "doctor",
}

# Words that describe the type of unit, not its name.
UNIT_WORDS = r"\b(city|municipality|province|of|the|island|islands)\b"


def strip_accents(text: str) -> str:
    """Remove accents, including ñ -> n, for comparison only."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize_name(name: str | None, drop_unit_words: bool = True) -> str:
    if not name:
        return ""
    s = strip_accents(str(name)).lower()
    s = re.sub(r"\(.*?\)", " ", s)          # parenthetical notes: (Capital), (Pob.)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)      # punctuation, hyphens, periods
    words = [ABBREVIATIONS.get(w, w) for w in s.split()]
    s = " ".join(w for w in words if w)
    if drop_unit_words:
        s = re.sub(UNIT_WORDS, " ", s)
    return re.sub(r"\s+", " ", s).strip()


ABSTAIN_PATTERNS = re.compile(
    r"^(unknown|n/?a|none|null|not sure|unsure|i (do not|don't) know|no idea|"
    r"cannot (determine|answer)|insufficient information)\.?$")


def is_abstention(answer) -> bool:
    if answer is None:
        return True
    if isinstance(answer, (list, tuple)):
        return len(answer) == 0 or all(is_abstention(a) for a in answer)
    text = str(answer).strip().lower()
    return text == "" or bool(ABSTAIN_PATTERNS.match(text))
