"""Day 4 analysis: who gets wrong answers?

Main question: do models know WHERE a town is (province and region questions)
less often for smaller, poorer, more rural, Visayas, or Mindanao towns?

Questions about places with recent official changes (Negros Island Region,
Sulu, Maguindanao) are excluded from the fairness tests, because those changes
are concentrated in the Visayas and Mindanao and would distort the comparison.

Outputs go to results/analysis/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bayanbench.stats import logit_cluster, wilson  # noqa: E402

MODELS = ["llama3.1-8b", "sealion-v3-8b", "gemma4-12b", "qwen3.5-9b"]
OUT = ROOT / "results" / "analysis"
INCOME = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}


def load():
    lgu = pd.read_csv(ROOT / "data/processed/lgus.csv", dtype={"code": str})
    lgu["log10_pop"] = np.log10(lgu.pop2024)
    lgu["income_num"] = lgu.income_class.map(INCOME)
    lgu["pop_quintile"] = pd.qcut(lgu.pop2024.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    frames = []
    for m in MODELS:
        d = pd.read_csv(ROOT / f"results/core/results_{m}_core_scored.csv", dtype={"lgu_code": str})
        d["model"] = m
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.merge(lgu[["code", "pop2024", "log10_pop", "pop_quintile", "income_class", "income_num",
                       "island_group", "urban_share", "unit_type", "is_ncr"]],
                  left_on="lgu_code", right_on="code", how="left")
    df["correct"] = (df.outcome == "correct").astype(int)
    # "Knows where the town is": an official former name (outdated) still shows the location.
    df["knows_location"] = df.outcome.isin(["correct", "outdated"]).astype(int)
    df["confident_wrong"] = df.outcome.isin(["wrong", "unreadable"]).astype(int)
    return lgu, df


def rate_table(d, by, value="correct"):
    rows = []
    for keys, g in d.groupby(by, observed=True):
        k, n = int(g[value].sum()), len(g)
        lo, hi = wilson(k, n)
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(dict(zip(by, keys), n=n, rate=100 * k / n, low=100 * lo, high=100 * hi))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lgu, df = load()
    loc = df[df.qtype.isin(["province", "region"]) & df.recent_change.isna()].copy()
    loc["correct"] = loc["knows_location"]   # location tests count correct or outdated answers
    report = {}

    # ---- A. population quintiles, island groups, income classes (location questions)
    by_q = rate_table(loc, ["model", "pop_quintile"])
    by_island = rate_table(loc, ["model", "island_group"])
    by_income = rate_table(loc.dropna(subset=["income_class"]), ["model", "income_class"])
    by_q.to_csv(OUT / "location_by_population_quintile.csv", index=False)
    by_island.to_csv(OUT / "location_by_island_group.csv", index=False)
    by_income.to_csv(OUT / "location_by_income_class.csv", index=False)

    # ---- B. regression with controls, clustered by town
    r = loc.dropna(subset=["income_num", "urban_share"]).copy()
    pop_sd = r.log10_pop.std()
    r["pop_z"] = (r.log10_pop - r.log10_pop.mean()) / pop_sd
    cols = {"const": np.ones(len(r)),
            "population (+1 SD of log10)": r.pop_z,
            "income class (+1 step poorer)": r.income_num - 1,
            "Visayas (vs Luzon)": (r.island_group == "Visayas").astype(float),
            "Mindanao (vs Luzon)": (r.island_group == "Mindanao").astype(float),
            "urban share (0 to 1)": r.urban_share,
            "city (vs municipality)": (r.unit_type == "city").astype(float),
            "region question (vs province)": (r.qtype == "region").astype(float)}
    for m in MODELS[1:]:
        cols[f"model: {m} (vs llama3.1-8b)"] = (r.model == m).astype(float)
    X = np.column_stack(list(cols.values()))
    rows, _, _ = logit_cluster(X, r.correct.values, r.lgu_code.values, list(cols))
    reg = pd.DataFrame(rows)
    reg.to_csv(OUT / "location_regression.csv", index=False)
    report["regression_n"] = int(len(r))
    report["regression_towns"] = int(r.lgu_code.nunique())
    report["pop_sd_log10"] = float(pop_sd)
    corr = lgu[["log10_pop", "income_num", "urban_share"]].corr().round(2)
    corr.to_csv(OUT / "predictor_correlations.csv")

    # Population effect within each model (simple, no other controls) for robustness
    per_model = []
    for m in MODELS:
        s = r[r.model == m]
        Xm = np.column_stack([np.ones(len(s)), s.pop_z, (s.qtype == "region").astype(float)])
        rm, _, _ = logit_cluster(Xm, s.correct.values, s.lgu_code.values, ["const", "pop", "region_q"])
        per_model.append({"model": m, "odds_ratio_per_sd_pop": rm[1]["odds_ratio"],
                          "low": rm[1]["or_low"], "high": rm[1]["or_high"], "p": rm[1]["p"]})
    pd.DataFrame(per_model).to_csv(OUT / "population_effect_by_model.csv", index=False)

    # ---- C. confident wrong vs saying unknown (location questions)
    hall = rate_table(loc, ["model", "pop_quintile"], value="confident_wrong")
    hall.to_csv(OUT / "location_confident_wrong_by_quintile.csv", index=False)

    # ---- D. barangays
    nb = df[df.qtype == "name_barangays"].copy()
    bq = nb.groupby(["model", "pop_quintile"]).agg(
        named=("n_named", "sum"), real_specific=("n_real_specific", "sum"),
        nowhere=("n_nowhere", "sum"), elsewhere=("n_elsewhere", "sum"),
        generic=("n_generic_wrong", "sum")).reset_index()
    bq["pct_real_specific"] = 100 * bq.real_specific / bq.named
    bq["pct_nowhere"] = 100 * bq.nowhere / bq.named
    bq.to_csv(OUT / "barangays_by_population_quintile.csv", index=False)
    bm = nb.groupby("model").agg(named=("n_named", "sum"), real_specific=("n_real_specific", "sum"),
                                 generic_wrong=("n_generic_wrong", "sum"), elsewhere=("n_elsewhere", "sum"),
                                 nowhere=("n_nowhere", "sum")).reset_index()
    for c in ["real_specific", "generic_wrong", "elsewhere", "nowhere"]:
        bm[f"pct_{c}"] = 100 * bm[c] / bm.named
    bm.to_csv(OUT / "barangays_by_model.csv", index=False)

    # ---- E. barangay count vs the "always guess the median" baseline, by quintile
    bc = df[df.qtype == "barangay_count"].copy()
    gold = bc.gold.astype(int)
    med = int(lgu.n_barangays.median())
    bc["baseline_within"] = (abs(gold - med) <= (gold * 0.10).round().clip(lower=1)).astype(int)
    bc["model_within"] = bc.within_10pct.fillna(False).astype(int)
    cnt = bc.groupby(["model", "pop_quintile"])[["model_within", "baseline_within"]].mean().mul(100).reset_index()
    cnt.to_csv(OUT / "count_vs_baseline_by_quintile.csv", index=False)
    report["median_barangays_national"] = med

    # ---- F. outdated answers on recently changed places
    rc = df[df.qtype.isin(["province", "region"]) & df.recent_change.notna()]
    out_tab = rc.groupby("model").outcome.value_counts().unstack(fill_value=0)
    out_tab.to_csv(OUT / "recent_change_outcomes.csv")

    # ---- G. overall table with confidence intervals
    overall = rate_table(df, ["model", "qtype"])
    overall.to_csv(OUT / "overall_accuracy.csv", index=False)

    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    pd.set_option("display.width", 200)
    print("LOCATION ACCURACY BY POPULATION QUINTILE (1 = smallest towns)")
    print(by_q.pivot(index="model", columns="pop_quintile", values="rate").round(1).to_string())
    print("\nBY ISLAND GROUP")
    print(by_island.pivot(index="model", columns="island_group", values="rate").round(1).to_string())
    print("\nBY INCOME CLASS (1st = richest)")
    print(by_income.pivot(index="model", columns="income_class", values="rate").round(1).to_string())
    print(f"\nREGRESSION (n={len(r):,} answers, {r.lgu_code.nunique()} towns, clustered by town)")
    print(reg[["term", "odds_ratio", "or_low", "or_high", "p"]].round(3).to_string(index=False))
    print("\nPOPULATION EFFECT WITHIN EACH MODEL (odds ratio per +1 SD of log population)")
    print(pd.DataFrame(per_model).round(3).to_string(index=False))
    print("\nPREDICTOR CORRELATIONS\n", corr.to_string())
    print("\nBARANGAYS BY MODEL")
    print(bm.round(1).to_string(index=False))
    print("\nSPECIFIC REAL BARANGAYS (% of names) BY POPULATION QUINTILE")
    print(bq.pivot(index="model", columns="pop_quintile", values="pct_real_specific").round(1).to_string())
    print("\nBARANGAY COUNT WITHIN 10%: MODEL vs BASELINE BY QUINTILE")
    print(cnt.pivot(index="model", columns="pop_quintile", values="model_within").round(1).to_string())
    print("baseline:", cnt.groupby("pop_quintile").baseline_within.first().round(1).to_dict())
    print("\nRECENTLY CHANGED PLACES (province and region questions)")
    print(out_tab.to_string())


if __name__ == "__main__":
    main()
