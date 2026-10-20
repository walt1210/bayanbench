"""Day 5: compare answers without and with PSGC records (core sample).

Needs, for each model that was run with records:
    results/core/results_{model}_core_scored.csv        (Day 3, no records)
    results/core_rag/results_{model}_core_rag_scored.csv (Day 5, with records)
Score the Day 5 files first: python scripts/score_results.py results/core_rag/results_{model}_core_rag.jsonl

Outputs go to results/analysis/rag/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from analyze import INCOME, MODELS  # noqa: E402
from bayanbench.stats import wilson  # noqa: E402

OUT = ROOT / "results" / "analysis" / "rag"


def load_pairs():
    lgu = pd.read_csv(ROOT / "data/processed/lgus.csv", dtype={"code": str})
    lgu["pop_quintile"] = pd.qcut(lgu.pop2024.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    lgu["income_num"] = lgu.income_class.map(INCOME)
    frames = []
    for m in MODELS:
        after_p = ROOT / f"results/core_rag/results_{m}_core_rag_scored.csv"
        if not after_p.exists():
            continue
        before = pd.read_csv(ROOT / f"results/core/results_{m}_core_scored.csv", dtype={"lgu_code": str})
        after = pd.read_csv(after_p, dtype={"lgu_code": str})
        keep = ["qid", "lgu_code", "qtype", "recent_change", "outcome", "within_10pct",
                "n_named", "n_real_specific", "n_nowhere", "n_elsewhere", "n_generic_wrong"]
        d = before[keep].merge(after[keep[0:1] + keep[4:]], on="qid", suffixes=("_before", "_after"))
        d["model"] = m
        frames.append(d)
    if not frames:
        sys.exit("No Day 5 scored files found in results/core_rag/.")
    df = pd.concat(frames, ignore_index=True)
    df = df.merge(lgu[["code", "pop_quintile", "island_group", "income_class"]],
                  left_on="lgu_code", right_on="code", how="left")
    for s in ("before", "after"):
        df[f"ok_{s}"] = (df[f"outcome_{s}"] == "correct").astype(int)
        df[f"loc_{s}"] = df[f"outcome_{s}"].isin(["correct", "outdated"]).astype(int)
    return df


def ci_text(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f} ({100 * lo:.1f} to {100 * hi:.1f})"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_pairs()
    models = list(df.model.unique())

    # A. accuracy by question type, with answers fixed and broken by the records
    rows = []
    for (m, qt), g in df.groupby(["model", "qtype"]):
        rows.append({"model": m, "qtype": qt, "n": len(g),
                     "before": 100 * g.ok_before.mean(), "after": 100 * g.ok_after.mean(),
                     "after_ci": ci_text(g.ok_after.sum(), len(g)),
                     "fixed": int(((g.ok_before == 0) & (g.ok_after == 1)).sum()),
                     "broken": int(((g.ok_before == 1) & (g.ok_after == 0)).sum()),
                     "unknown_after": int((g.outcome_after == "abstained").sum())})
    acc = pd.DataFrame(rows)
    acc["change"] = acc.after - acc.before
    acc.to_csv(OUT / "accuracy_before_after.csv", index=False)

    # B. location questions (excluding recent changes) by population quintile: does the gap close?
    loc = df[df.qtype.isin(["province", "region"]) & df.recent_change.isna()]
    q = loc.groupby(["model", "pop_quintile"])[["loc_before", "loc_after"]].mean().mul(100).reset_index()
    q.to_csv(OUT / "location_by_quintile_before_after.csv", index=False)
    gap = []
    for m in models:
        s = q[q.model == m].set_index("pop_quintile")
        gap.append({"model": m, "gap_before": s.loc[5, "loc_before"] - s.loc[1, "loc_before"],
                    "gap_after": s.loc[5, "loc_after"] - s.loc[1, "loc_after"],
                    "smallest_before": s.loc[1, "loc_before"], "smallest_after": s.loc[1, "loc_after"]})
    gap = pd.DataFrame(gap)
    gap.to_csv(OUT / "location_gap_before_after.csv", index=False)
    isl = loc.groupby(["model", "island_group"])[["loc_before", "loc_after"]].mean().mul(100).reset_index()
    isl.to_csv(OUT / "location_by_island_before_after.csv", index=False)

    # C. recently changed places: do records replace outdated answers?
    rc = df[df.qtype.isin(["province", "region"]) & df.recent_change.notna()]
    rct = rc.groupby("model")[["outcome_before", "outcome_after"]].agg(
        lambda s: s.value_counts().to_dict())
    rct.to_csv(OUT / "recent_change_before_after.csv")

    # D. barangay names
    nb = df[df.qtype == "name_barangays"]
    b = nb.groupby("model").agg(
        named_before=("n_named_before", "sum"), specific_before=("n_real_specific_before", "sum"),
        nowhere_before=("n_nowhere_before", "sum"),
        named_after=("n_named_after", "sum"), specific_after=("n_real_specific_after", "sum"),
        nowhere_after=("n_nowhere_after", "sum")).reset_index()
    for s in ("before", "after"):
        b[f"pct_specific_{s}"] = 100 * b[f"specific_{s}"] / b[f"named_{s}"]
        b[f"pct_made_up_{s}"] = 100 * b[f"nowhere_{s}"] / b[f"named_{s}"]
    b.to_csv(OUT / "barangays_before_after.csv", index=False)

    # E. barangay count within 10%
    bc = df[df.qtype == "barangay_count"].copy()
    for s in ("before", "after"):
        bc[f"w_{s}"] = bc[f"within_10pct_{s}"].fillna(False).astype(bool)
    cnt = bc.groupby("model")[["w_before", "w_after"]].mean().mul(100).reset_index()
    cnt.to_csv(OUT / "count_within10_before_after.csv", index=False)

    pd.set_option("display.width", 200)
    print("ACCURACY BEFORE -> AFTER (percent correct; fixed and broken are answer counts)")
    print(acc[["model", "qtype", "n", "before", "after", "change", "fixed", "broken", "unknown_after"]]
          .round(1).to_string(index=False))
    print("\nLOCATION (correct or outdated) BY POPULATION QUINTILE")
    print(q.pivot(index="model", columns="pop_quintile", values=["loc_before", "loc_after"]).round(1).to_string())
    print("\nGAP: LARGEST minus SMALLEST QUINTILE (percentage points)")
    print(gap.round(1).to_string(index=False))
    print("\nBY ISLAND GROUP")
    print(isl.pivot(index="model", columns="island_group", values=["loc_before", "loc_after"]).round(1).to_string())
    print("\nRECENTLY CHANGED PLACES")
    print(rct.to_string())
    print("\nBARANGAY NAMES")
    print(b[["model", "pct_specific_before", "pct_specific_after", "pct_made_up_before", "pct_made_up_after"]]
          .round(1).to_string(index=False))
    print("\nBARANGAY COUNT WITHIN 10%")
    print(cnt.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
