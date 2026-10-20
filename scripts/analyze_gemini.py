"""Gemini 3.6 Flash (batched, 25 questions per request) on the core sample.

Gemini is only run without records (free-tier limits), so it is compared with the
open models' no-lookup answers. Outputs: results/analysis/gemini/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bayanbench.stats import logit_cluster  # noqa: E402

NAME = "gemini-3.6-flash-batch25"
OUT = ROOT / "results/analysis/gemini"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lgu = pd.read_csv(ROOT / "data/processed/lgus.csv", dtype={"code": str})
    lgu["pop_quintile"] = pd.qcut(lgu.pop2024.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    d = pd.read_csv(ROOT / f"results/core/results_{NAME}_core_scored.csv", dtype={"lgu_code": str})
    d = d.merge(lgu[["code", "pop2024", "pop_quintile", "island_group"]], left_on="lgu_code", right_on="code")
    d["correct"] = d.outcome == "correct"
    d["knows_location"] = d.outcome.isin(["correct", "outdated"])

    overall = d.groupby("qtype").outcome.value_counts(normalize=True).mul(100).unstack(fill_value=0).round(1)
    overall.to_csv(OUT / "overall.csv")
    loc = d[d.qtype.isin(["province", "region"]) & d.recent_change.isna()]
    loc.groupby("pop_quintile").knows_location.mean().mul(100).round(1).to_csv(OUT / "location_by_quintile.csv")

    nb = d[d.qtype == "name_barangays"]
    agg = {"named": ("n_named", "sum"), "specific": ("n_real_specific", "sum"), "made_up": ("n_nowhere", "sum"),
           "elsewhere": ("n_elsewhere", "sum"), "all_three_real": ("correct", "mean")}
    for by in ("pop_quintile", "island_group"):
        t = nb.groupby(by).agg(**agg)
        t["pct_specific"] = 100 * t.specific / t.named
        t["pct_made_up"] = 100 * t.made_up / t.named
        t["all_three_real"] *= 100
        t.round(1).to_csv(OUT / f"barangays_by_{by}.csv")

    # Town size and island group vs. naming a real barangay of that town (one row per name).
    rows = []
    for r in nb.itertuples():
        n, k = int(r.n_named), int(r.n_real_specific)
        rows += [(r.lgu_code, np.log10(r.pop2024), r.island_group, 1)] * k
        rows += [(r.lgu_code, np.log10(r.pop2024), r.island_group, 0)] * (n - k)
    x = pd.DataFrame(rows, columns=["town", "log_pop", "island", "y"])
    z = (x.log_pop - x.log_pop.mean()) / x.log_pop.std()
    X = np.column_stack([np.ones(len(x)), z, x.island == "Visayas", x.island == "Mindanao"]).astype(float)
    reg, _, _ = logit_cluster(X, x.y.values, x.town.values,
                              ["const", "population (+1 SD of log10)", "Visayas (vs Luzon)", "Mindanao (vs Luzon)"])
    pd.DataFrame(reg).to_csv(OUT / "barangay_name_regression.csv", index=False)

    # Does answer quality drift within a batch of 25?
    pos = {}
    for line in open(ROOT / f"results/core/results_{NAME}_core.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("error") is None:
            pos[r["qid"]] = r.get("batch_position")
    d["batch_position"] = d.qid.map(pos)
    drift = d.groupby(pd.cut(d.batch_position, [-1, 7, 15, 24], labels=["1-8", "9-16", "17-25"]),
                      observed=True).correct.mean().mul(100).round(1)
    drift.to_csv(OUT / "accuracy_by_batch_position.csv")

    print(overall.to_string())
    print("\nnon-correct count and region answers:")
    print(d[d.qtype.isin(["barangay_count", "region"]) & ~d.correct][["lgu_name", "qtype", "answer", "gold", "outcome"]]
          .to_string(index=False))
    print("\nbarangays by town size:\n", pd.read_csv(OUT / "barangays_by_pop_quintile.csv").to_string(index=False))
    print("\nregression (one row per name):\n", pd.DataFrame(reg)[["term", "odds_ratio", "or_low", "or_high", "p"]]
          .round(4).to_string(index=False))
    print("\naccuracy by position in batch:", drift.to_dict())


if __name__ == "__main__":
    main()
