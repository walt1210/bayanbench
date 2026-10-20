"""Write notebooks/04_ask_open_models_rag.ipynb (Day 5: answers with PSGC records).

Same models, settings and answer rules as 02_ask_open_models.ipynb. The only
change: each question is preceded by up to three official PSGC records found
by the retriever (scripts/build_rag_questions.py), and the instruction says to
use them.
"""
import json
from pathlib import Path

import make_notebook_open as base

ROOT = Path(__file__).resolve().parents[1]

OLD_SYSTEM = '''SYSTEM = """You answer factual questions about places in the Philippines.
Reply with JSON only'''
NEW_SYSTEM = '''SYSTEM = """You answer factual questions about places in the Philippines.
Before each question you are given reference records from the official PSGC list
(Philippine Statistics Authority). Use them when they are about the place in the question.
Reply with JSON only'''
assert OLD_SYSTEM in base.CLIENT
CLIENT = base.CLIENT.replace(OLD_SYSTEM, NEW_SYSTEM)
assert 'ask(tag, info, q["question"])' in CLIENT
CLIENT = CLIENT.replace('ask(tag, info, q["question"])', 'ask(tag, info, q["prompt"])')

OLD_UNLOAD = CLIENT[CLIENT.index("def unload(tag):"):CLIENT.index("def parse(text):")]
NEW_UNLOAD = '''def unload(tag, wait=120):
    """Remove the model from memory and wait until Ollama confirms it is gone."""
    try:
        requests.post(f"{OLLAMA}/api/generate", json={"model": tag, "keep_alive": 0}, timeout=60)
        for _ in range(wait):
            loaded = requests.get(f"{OLLAMA}/api/ps", timeout=10).json().get("models", [])
            if not loaded:
                return
            time.sleep(1)
    except requests.RequestException:
        pass


'''
CLIENT = CLIENT.replace(OLD_UNLOAD, NEW_UNLOAD)

RUN_CELL = """
def progress():
    rows = {}
    for name in MODELS:
        p = f"{SAVE_DIR}/results_{name}_{SAMPLE}_rag.jsonl"
        rows[name] = len(done_ids(p)) if os.path.exists(p) else 0
    return rows

print("Answered so far:", {k: f"{v}/{len(questions)}" for k, v in progress().items()})
summary = {}
for name, tag in MODELS.items():
    results_file = f"{SAVE_DIR}/results_{name}_{SAMPLE}_rag.jsonl"
    if os.path.exists(results_file) and len(done_ids(results_file)) >= len(questions):
        print(f"{name}: already finished, skipped")
        continue
    if RUN_ONLY and name != RUN_ONLY:
        continue
    print(f"\\n=== {name} ({tag}) ===")
    t0 = time.time()
    pull(tag)
    print(f"downloaded in {time.time() - t0:.0f}s")
    info = run_model(name, tag, questions, results_file, workers=WORKERS)
    summary[name] = dict(summarize(results_file, len(questions)), quantization=info["quantization"])
    unload(tag)
    !ollama rm {tag} > /dev/null 2>&1
    gc.collect()

import pandas as pd
print("Answered so far:", {k: f"{v}/{len(questions)}" for k, v in progress().items()})
pd.DataFrame(summary).T
"""

md, code = base.md, base.code
cells = []
for c in base.CELLS:
    src = "".join(c["source"])
    if src.startswith("# BayanBench: Ask Open Models"):
        c = md("""
# BayanBench Day 5: Ask Open Models With PSGC Records

Same models and settings as notebook 02. The only difference: before each question, the model is shown up to three **official PSGC records** (province, region, city or municipality, number of barangays, barangay names) found from the question text.

**Before running:** Runtime → Change runtime type → **T4 GPU**.

Then Runtime → Run all. When asked, upload `questions_core_rag.jsonl`. Progress is saved after every answer; if Colab disconnects, run all cells again and it continues.
""")
    elif c["cell_type"] == "code" and src.startswith("SAMPLE ="):
        src = src.replace('SAMPLE = "pilot"   # "pilot" (232 questions) or "core" (1,825 questions)',
                          'SAMPLE = "core"    # "pilot" (232 questions) or "core" (1,825 questions)')
        src = src.replace("WORKERS = 4        # questions answered at the same time",
                          "WORKERS = 2        # questions answered at the same time (2 uses less memory)\n"
                          "# Run one model per session to avoid running out of RAM, e.g. RUN_ONLY = \"gemma4-12b\".\n"
                          "# None runs every unfinished model in order. Finished models are always skipped.\n"
                          "RUN_ONLY = None")
        assert "RUN_ONLY" in src
        c = code(src)
    elif c["cell_type"] == "code" and "OLLAMA_NUM_PARALLEL" in src:
        assert '"4"' in src
        c = code(src.replace('os.environ["OLLAMA_NUM_PARALLEL"] = "4"', 'os.environ["OLLAMA_NUM_PARALLEL"] = "2"')
                 .replace("import os, subprocess, time, requests", "import os, subprocess, time, requests, gc"))
    elif c["cell_type"] == "code" and "QUESTIONS_FILE" in src:
        c = code(src.replace("questions_{SAMPLE}.jsonl", "questions_{SAMPLE}_rag.jsonl"))
    elif c["cell_type"] == "code" and src.lstrip().startswith("import json, os, re"):
        c = code(CLIENT)
    elif c["cell_type"] == "code" and src.startswith("summary = {}"):
        c = code(RUN_CELL)
    elif c["cell_type"] == "code" and ("results_{name}_{SAMPLE}.jsonl" in src):
        c = code(src.replace("results_{name}_{SAMPLE}.jsonl", "results_{name}_{SAMPLE}_rag.jsonl"))
    cells.append(c)


def main():
    nb = {"cells": cells, "metadata": {"colab": {"provenance": [], "gpuType": "T4"},
          "accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    out = ROOT / "notebooks" / "04_ask_open_models_rag.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "notebooks" / "open_client_rag.py").write_text(CLIENT.lstrip(), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
