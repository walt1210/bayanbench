"""Look up official PSGC records for the place named in a question.

The retriever sees only the question text. It does not know which town the
question is about, so it can fail the way a real lookup would (for example,
when two towns share a name and the question gives no province).

How it works:
1. Every city and municipality is indexed by its normalized name.
2. Every run of 1 to 6 consecutive words in the normalized question is looked
   up in that index. A match inside a longer match is dropped, so
   "San Jose del Monte" does not also return "San Jose".
3. Matching towns are ranked: towns whose province or region is also named in
   the question first, then longer names, then larger population. This puts
   Moncada first for "Moncada, Tarlac", not Tarlac City.
4. Up to K records are returned as short text, with the barangay list cut to
   MAX_BARANGAYS names (in official PSGC code order).
"""
from collections import defaultdict

import pandas as pd

from .normalize import normalize_name

K = 3
MAX_BARANGAYS = 40
MAX_NGRAM = 6


def _norm(text):
    return normalize_name(text, drop_unit_words=False)


class PSGCRetriever:
    def __init__(self, lgus: pd.DataFrame, barangays: pd.DataFrame):
        self.lgus = lgus.set_index("code", drop=False)
        self.index = defaultdict(list)
        for r in lgus.itertuples():
            self.index[_norm(r.display_name)].append(r.code)
        self.bgy = {c: list(g.sort_values("code").name)
                    for c, g in barangays.groupby("lgu_code")}

    def search(self, question: str, k: int = K) -> list[str]:
        words = _norm(question).split()
        spans = []  # (start, end, phrase) of every word run that is a town name
        for n in range(min(MAX_NGRAM, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                if phrase in self.index:
                    spans.append((i, i + n, phrase))
        # Drop a match that sits inside a longer match ("San Jose" in "San Jose del Monte").
        names = {p for i, j, p in spans
                 if not any(a <= i and j <= b and (b - a) > (j - i) for a, b, _ in spans)}
        if not names:
            return []
        q = " " + " ".join(words) + " "
        codes = {c for nm in names for c in self.index[nm]}

        def rank(code):
            r = self.lgus.loc[code]
            ctx = [x for x in (r.province, r.region) if isinstance(x, str)]
            # The province must appear somewhere other than the town name itself,
            # so "Moncada, Tarlac" ranks Moncada above Tarlac City.
            rest = q.replace(f" {_norm(r.display_name)} ", " ", 1)
            ctx_hit = any(f" {_norm(x)} " in rest for x in ctx)
            return (not ctx_hit, -len(_norm(r.display_name).split()),
                    -(r.pop2024 if pd.notna(r.pop2024) else 0))

        return sorted(codes, key=rank)[:k]

    def record(self, code: str) -> str:
        r = self.lgus.loc[code]
        if r.unit_type == "city":
            kind = {"HUC": "highly urbanized city", "ICC": "independent component city",
                    "CC": "component city"}.get(r.city_class, "city")
        else:
            kind = "municipality"
        where = (f"in the province of {r.province}" if isinstance(r.province, str)
                 else "not under any province")
        names = self.bgy.get(code, [])
        shown = ", ".join(names[:MAX_BARANGAYS])
        more = f", and {len(names) - MAX_BARANGAYS} more" if len(names) > MAX_BARANGAYS else ""
        return (f"{r['name']}: a {kind} (not a municipality)" if r.unit_type == "city"
                else f"{r['name']}: a municipality (not a city)") + \
            (f", {where}, region: {r.region}. "
             f"Number of barangays: {len(names)}. Barangays: {shown}{more}.")

    def context(self, question: str, k: int = K) -> tuple[str, list[str]]:
        codes = self.search(question, k)
        if not codes:
            return "No matching record was found.", []
        text = "\n".join(f"[{i}] {self.record(c)}" for i, c in enumerate(codes, 1))
        return text, codes
