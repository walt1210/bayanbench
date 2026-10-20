"""Day 5: add official PSGC records to each question (retrieval-augmented version).

The question text is unchanged. Each question gets a "prompt" that starts with
up to three PSGC records found by the retriever from the question text alone.

Outputs:
  data/processed/questions_{sample}_rag.jsonl   for the Colab notebook
  results/analysis/retrieval_{sample}.csv       whether the right town was retrieved

Usage: python scripts/build_rag_questions.py core
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bayanbench.retrieve import PSGCRetriever  # noqa: E402

PROMPT = "Reference records from the official PSGC list:\n{context}\n\nQuestion: {question}"


def main(sample="core"):
    lgus = pd.read_csv(ROOT / "data/processed/lgus.csv", dtype={"code": str})
    bgy = pd.read_csv(ROOT / "data/processed/barangays.csv", dtype={"code": str, "lgu_code": str})
    ret = PSGCRetriever(lgus, bgy)
    qs = [json.loads(l) for l in open(ROOT / f"data/processed/questions_{sample}.jsonl", encoding="utf-8")]
    out, stats = [], []
    for q in qs:
        ctx, codes = ret.context(q["question"])
        out.append(dict(q, prompt=PROMPT.format(context=ctx, question=q["question"])))
        stats.append({"qid": q["qid"], "lgu_code": q["lgu_code"], "qtype": q["qtype"],
                      "top1_hit": bool(codes) and codes[0] == q["lgu_code"],
                      "topk_hit": q["lgu_code"] in codes, "n_records": len(codes),
                      "prompt_chars": len(out[-1]["prompt"])})
    p = ROOT / f"data/processed/questions_{sample}_rag.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    s = pd.DataFrame(stats)
    (ROOT / "results/analysis").mkdir(parents=True, exist_ok=True)
    s.to_csv(ROOT / f"results/analysis/retrieval_{sample}.csv", index=False)
    print(f"wrote {p.name}: {len(out)} questions")
    print(f"right town ranked first: {s.top1_hit.mean():.1%}; in the records shown: {s.topk_hit.mean():.1%}")
    print(s.groupby("qtype")[["top1_hit", "topk_hit"]].mean().round(3).to_string())
    print(f"prompt length: median {s.prompt_chars.median():.0f} chars, max {s.prompt_chars.max()}")
    return s


if __name__ == "__main__":
    main(*sys.argv[1:])
