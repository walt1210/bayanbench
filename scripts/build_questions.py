"""Generate benchmark questions from the cleaned PSGC tables.

Every question has an official answer from PSGC. Five question types:

    province        Which province is X in?                   (unique names only, not independent cities)
    region          Which region is X in?                     (unique names only)
    unit_type       Is X, <province> a city or a municipality?
    barangay_count  How many barangays are in X, <province>?
    name_barangays  Name three barangays in X, <province>.

Questions avoid revealing the answer: city names are shown without the word
"City", and location questions do not mention the province or region.

Also writes two samples for running within free API limits:
    pilot   50 towns, for debugging prompts and the scorer (Day 2)
    core    400 towns, stratified by island group and population quintile

Usage:
    python scripts/build_questions.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SEED = 2026

TEMPLATES = {
    "province": "In which province of the Philippines is {name} (a city or municipality) located?",
    "region": "In which administrative region of the Philippines is {name} (a city or municipality) located?",
    "unit_type": "Is {name}, {context}, Philippines, a city or a municipality?",
    "barangay_count": "How many barangays are in {name}, {context}, Philippines?",
    "name_barangays": "Name three barangays in {name}, {context}, Philippines.",
}


def build(lgu: pd.DataFrame) -> list[dict]:
    qs = []
    for r in lgu.itertuples():
        base = {"lgu_code": r.code, "lgu_name": r.name, "display_name": r.display_name,
                "context": r.context, "recent_change": r.recent_change if isinstance(r.recent_change, str) else None}

        def add(qtype, gold, scoring, **extra):
            q = dict(base, qid=f"{r.code}-{qtype}", qtype=qtype, scoring=scoring,
                     question=TEMPLATES[qtype].format(name=r.display_name, context=r.context),
                     gold=gold)
            q.update(extra)
            qs.append(q)

        if not r.ambiguous_name and not r.independent_city and not r.new_lgu:
            add("province", [r.province], "province",
                outdated=[r.previous_province] if isinstance(r.previous_province, str) else None)
        if not r.ambiguous_name and not r.new_lgu:
            add("region", json.loads(r.region_options), "place",
                outdated=json.loads(r.prev_region_options) if isinstance(r.prev_region_options, str) else None)
        add("unit_type", [r.unit_type], "unit_type")
        add("barangay_count", int(r.n_barangays), "count")
        add("name_barangays", {"lgu_code": r.code, "n_official": int(r.n_barangays)}, "barangays")
    return qs


def stratified_sample(lgu: pd.DataFrame, n: int, seed: int) -> list[str]:
    """Equal share per island group, spread evenly across national population quintiles."""
    pool = lgu[~lgu.new_lgu].copy()
    pool["pop_q"] = pd.qcut(pool.pop2024.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rng = np.random.default_rng(seed)
    cells = pool.groupby(["island_group", "pop_q"], observed=True)
    per_cell = n // cells.ngroups
    picked = []
    for _, cell in cells:
        k = min(per_cell, len(cell))
        picked += list(rng.choice(cell.code.values, size=k, replace=False))
    rest = pool[~pool.code.isin(picked)]
    picked += list(rng.choice(rest.code.values, size=n - len(picked), replace=False))
    return sorted(picked)


def main():
    lgu = pd.read_csv(PROC / "lgus.csv", dtype={"code": str, "region_code": str})
    qs = build(lgu)
    with open(PROC / "questions.jsonl", "w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    core = stratified_sample(lgu, 400, SEED)
    pilot = sorted(np.random.default_rng(SEED + 1).choice(core, size=50, replace=False))
    (PROC / "samples.json").write_text(json.dumps(
        {"seed": SEED, "pilot": list(pilot), "core": list(core)}, indent=1))

    # Question-only files for the notebooks: no answers, so models never see them.
    for name, codes in [("pilot", set(pilot)), ("core", set(core)), ("full", None)]:
        with open(PROC / f"questions_{name}.jsonl", "w", encoding="utf-8") as f:
            for q in qs:
                if codes is None or q["lgu_code"] in codes:
                    f.write(json.dumps({k: q[k] for k in ("qid", "lgu_code", "qtype", "question")},
                                       ensure_ascii=False) + "\n")

    df = pd.DataFrame(qs)
    print(f"{len(df):,} questions about {df.lgu_code.nunique():,} cities and municipalities")
    print(df.qtype.value_counts().to_string())
    print(f"questions about recently changed places: {df.recent_change.notna().sum()}")
    core_q = df[df.lgu_code.isin(core)]
    print(f"core sample: {len(core)} towns, {len(core_q):,} questions; pilot: {len(pilot)} towns, "
          f"{df.lgu_code.isin(pilot).sum()} questions")


if __name__ == "__main__":
    main()
