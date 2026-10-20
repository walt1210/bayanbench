"""Compare answers to the same questions in two separate runs (pilot towns are inside the core sample).

Usage: python scripts/reproducibility_check.py
Writes results/analysis/reproducibility.csv
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["llama3.1-8b", "sealion-v3-8b", "qwen3.5-9b", "gemma4-12b"]

rows = []
for m in MODELS:
    load = lambda p: {json.loads(l)["qid"]: json.loads(l) for l in open(p, encoding="utf-8")}
    p = load(ROOT / f"results/pilot/results_{m}_pilot.jsonl")
    c = load(ROOT / f"results/core/results_{m}_core.jsonl")
    sp = pd.read_csv(ROOT / f"results/pilot/results_{m}_pilot_scored.csv").set_index("qid")
    sc = pd.read_csv(ROOT / f"results/core/results_{m}_core_scored.csv").set_index("qid").loc[sp.index]
    rows.append({"model": m, "questions": len(p),
                 "identical_answer_pct": 100 * sum(json.dumps(p[q]["answer"]) == json.dumps(c[q]["answer"]) for q in p) / len(p),
                 "same_verdict_pct": 100 * (sp.outcome == sc.outcome).mean(),
                 "accuracy_run1": 100 * (sp.outcome == "correct").mean(),
                 "accuracy_run2": 100 * (sc.outcome == "correct").mean()})
df = pd.DataFrame(rows).round(1)
df.to_csv(ROOT / "results/analysis/reproducibility.csv", index=False)
print(df.to_string(index=False))
