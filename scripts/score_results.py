"""Score a results file from the notebook against the official PSGC answers.

Outputs, next to the results file:
    <name>_scored.csv       one row per question with the outcome and details
    <name>_handcheck.csv    a stratified random sample for checking the scorer by hand
and prints a summary table.

Outcomes: correct, outdated, abstained, wrong, unreadable (reply was not valid JSON).
For "name three barangays", each invented name is also classified as
    elsewhere   a real barangay, but in a different city or municipality
    nowhere     not a barangay anywhere in the Philippines

Usage:
    python scripts/score_results.py results/results_gemini-3.6-flash_pilot.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bayanbench.normalize import normalize_name  # noqa: E402
from bayanbench.score import (is_generic_barangay, score_barangays,  # noqa: E402
                              score_choice, score_count)

PROC = ROOT / "data" / "processed"


def latest_answers(path: Path) -> dict:
    """Keep the most recent successful reply per question (reruns append duplicates)."""
    latest = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("error") is None:
            latest[r["qid"]] = r
    return latest


def main(results_path: str, handcheck_per_type: int = 10, seed: int = 7) -> pd.DataFrame:
    results_path = Path(results_path)
    gold = {json.loads(l)["qid"]: json.loads(l) for l in open(PROC / "questions.jsonl", encoding="utf-8")}
    bgy = pd.read_csv(PROC / "barangays.csv", dtype={"lgu_code": str})
    by_lgu = bgy.groupby("lgu_code")["name"].apply(list).to_dict()
    national_keys = set(bgy.name.map(normalize_name))

    rows = []
    for qid, r in latest_answers(results_path).items():
        q = gold[qid]
        ans = r.get("answer")
        row = {"qid": qid, "lgu_code": q["lgu_code"], "lgu_name": q["lgu_name"], "qtype": q["qtype"],
               "question": q["question"], "model": r.get("model"),
               "model_version": r.get("model_version"), "answer": json.dumps(ans, ensure_ascii=False),
               "recent_change": q.get("recent_change")}
        if r.get("parse_error"):
            row.update(outcome="unreadable", gold=str(q["gold"]), detail=r.get("parse_error"))
        elif q["scoring"] in ("province", "place", "unit_type"):
            row.update(outcome=score_choice(ans, q["gold"], q["scoring"], q.get("outdated")),
                       gold=" / ".join(q["gold"]))
        elif q["scoring"] == "count":
            s = score_count(ans, q["gold"])
            row.update(outcome=s["outcome"], gold=str(q["gold"]),
                       within_10pct=s["within_tolerance"], detail=f"value={s.get('value')}")
        elif q["scoring"] == "barangays":
            official = by_lgu[q["gold"]["lgu_code"]]
            s = score_barangays(ans, official)
            generic_wrong = [n for n in s["invented"] if is_generic_barangay(n)]
            specific = [n for n in s["invented"] if not is_generic_barangay(n)]
            elsewhere = [n for n in specific if normalize_name(n) in national_keys]
            nowhere = [n for n in specific if normalize_name(n) not in national_keys]
            real_names = [m[0] for m in s["matches"]]
            real_specific = [n for n in real_names if not is_generic_barangay(n)]
            row.update(outcome=s["outcome"], gold=f"{len(official)} official barangays",
                       n_named=s["named"], n_real=s["real"], n_real_specific=len(real_specific),
                       n_generic_wrong=len(generic_wrong), n_elsewhere=len(elsewhere),
                       n_nowhere=len(nowhere), detail=json.dumps(
                           {"real": [m[1] for m in s["matches"]], "generic_wrong": generic_wrong,
                            "elsewhere": elsewhere, "nowhere": nowhere}, ensure_ascii=False))
        rows.append(row)

    df = pd.DataFrame(rows)
    stem = results_path.with_suffix("")
    df.to_csv(f"{stem}_scored.csv", index=False)

    # Hand-check sample: equal number per question type, fixed seed.
    rng = np.random.default_rng(seed)
    picks = []
    for _, g in df.groupby("qtype"):
        picks.append(g.iloc[rng.choice(len(g), size=min(handcheck_per_type, len(g)), replace=False)])
    hc = pd.concat(picks).reindex(
        columns=["qid", "qtype", "question", "gold", "answer", "outcome", "detail"])
    hc["your_verdict"] = ""   # write: agree / disagree, with a short reason
    hc.to_csv(f"{stem}_handcheck.csv", index=False)

    summary = (df.groupby("qtype").outcome.value_counts(normalize=True).unstack(fill_value=0) * 100).round(1)
    print(f"{len(df)} scored answers from {results_path.name}\n")
    print(summary.to_string())
    if "within_10pct" in df:
        wc = df[df.qtype == "barangay_count"].within_10pct.mean() * 100
        print(f"\nbarangay count within 10%: {wc:.1f}%")
    nb = df[df.qtype == "name_barangays"]
    if len(nb) and nb.n_named.sum():
        print(f"named barangays: {int(nb.n_named.sum())} | real: {int(nb.n_real.sum())} "
              f"(specific names: {int(nb.n_real_specific.sum())}) | wrong generic placeholder: "
              f"{int(nb.n_generic_wrong.sum())} | real but in another town: {int(nb.n_elsewhere.sum())} "
              f"| not found anywhere: {int(nb.n_nowhere.sum())}")
    print(f"\nwrote {stem}_scored.csv and {stem}_handcheck.csv")
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
