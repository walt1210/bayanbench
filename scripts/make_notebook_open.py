"""Write notebooks/02_ask_open_models.ipynb.

Runs open models on Colab's free GPU through Ollama, with the same
instruction and settings as the Gemini notebook. No daily limits.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


# The client code lives in one string so it can be tested outside Colab.
CLIENT = r'''
import json, os, re, time, datetime, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

SYSTEM = """You answer factual questions about places in the Philippines.
Reply with JSON only, in this format: {"answer": <your answer>}
- For a province or a region, give its name as a string.
- For "a city or a municipality", answer "city" or "municipality".
- For a number of barangays, give a whole number.
- For "name three barangays", give a list of three barangay names.
If you are not sure, set the answer to "unknown"."""

OPTIONS = {"temperature": 0.0, "seed": 2026, "num_ctx": 2048, "num_predict": 256}


def pull(tag):
    r = requests.post(f"{OLLAMA}/api/pull", json={"model": tag, "stream": False}, timeout=3600)
    r.raise_for_status()


def model_info(tag):
    """Exact version details recorded with every answer."""
    info = requests.post(f"{OLLAMA}/api/show", json={"model": tag}, timeout=60).json()
    details = info.get("details", {})
    thinking = info.get("thinking") or {}
    values = thinking.get("values")
    can_disable = values is None or False in values or "false" in [str(v).lower() for v in values]
    has_thinking = "thinking" in (info.get("capabilities") or []) or bool(values)
    return {"tag": tag, "digest": info.get("digest") or details.get("digest"),
            "quantization": details.get("quantization_level"),
            "parameters": details.get("parameter_size"), "family": details.get("family"),
            "has_thinking": has_thinking, "can_disable_thinking": can_disable}


def unload(tag):
    try:
        requests.post(f"{OLLAMA}/api/generate", json={"model": tag, "keep_alive": 0}, timeout=60)
    except requests.RequestException:
        pass


def parse(text):
    if not text:
        return None, "empty response"
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return None, "not JSON"
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, "not JSON"
    if isinstance(obj, dict) and "answer" in obj:
        return obj["answer"], None
    return None, "no 'answer' field"


def ask(tag, info, question, retries=3):
    body = {"model": tag, "stream": False, "format": "json", "options": dict(OPTIONS),
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": question}]}
    if info["has_thinking"]:
        if info["can_disable_thinking"]:
            body["think"] = False
        else:  # thinking cannot be switched off: leave room for it
            body["options"]["num_predict"] = 4096
    for attempt in range(retries + 1):
        try:
            r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=600)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(5 * (attempt + 1))


def done_ids(path):
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("error") is None:
                done.add(r["qid"])
    return done


def run_model(name, tag, questions, results_file, workers=4, log_every=25):
    info = model_info(tag)
    todo = [q for q in questions if q["qid"] not in done_ids(results_file)]
    print(f"{name}: {len(questions) - len(todo)} done, {len(todo)} to go | "
          f"quantization={info['quantization']} thinking={info['has_thinking']} "
          f"can_disable={info['can_disable_thinking']}")
    lock = threading.Lock()
    t0, n = time.time(), 0

    def one(q):
        rec = {"qid": q["qid"], "qtype": q["qtype"], "model": name, "model_tag": tag,
               "model_digest": info["digest"], "quantization": info["quantization"],
               "time": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            resp = ask(tag, info, q["question"])
            msg = resp.get("message", {})
            answer, perr = parse(msg.get("content"))
            rec.update(raw=msg.get("content"), answer=answer, parse_error=perr, error=None,
                       thinking_chars=len(msg.get("thinking") or ""),
                       done_reason=resp.get("done_reason"), eval_count=resp.get("eval_count"))
        except Exception as e:
            rec.update(raw=None, answer=None, parse_error=None, error=str(e)[:300])
        return q, rec

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one, q) for q in todo]):
            q, rec = fut.result()
            with lock:
                with open(results_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % log_every == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s)  "
                          f"{q['question'][:55]} -> {rec.get('answer')}")
    return info


def summarize(results_file, n_questions):
    latest = {}
    for line in open(results_file, encoding="utf-8"):
        r = json.loads(line)
        if r.get("error") is None:
            latest[r["qid"]] = r
    vals = list(latest.values())
    return {"answered": len(latest), "of": n_questions,
            "unreadable": sum(r.get("parse_error") is not None for r in vals),
            "unknown": sum(str(r.get("answer")).strip().lower() == "unknown" for r in vals),
            "cut_off": sum(r.get("done_reason") == "length" for r in vals)}
'''

CELLS = [
    md("""
# BayanBench: Ask Open Models

Runs open models on Colab's free GPU and saves their **raw answers**. The instruction and settings match the Gemini notebook exactly. Scoring happens separately.

**Before running:** Runtime → Change runtime type → **T4 GPU**.

Then run all cells (Runtime → Run all). When asked, upload `questions_pilot.jsonl` (or `questions_core.jsonl`). Progress is saved after every answer; if Colab disconnects, run all cells again and it continues.
"""),
    md("## 1. Settings"),
    code("""
SAMPLE = "pilot"   # "pilot" (232 questions) or "core" (1,825 questions)

# Models to run, in order. Comment out a line to skip a model.
MODELS = {
    "llama3.1-8b":   "llama3.1:8b",
    "sealion-v3-8b": "aisingapore/Llama-SEA-LION-v3-8B-IT:q4_k_m",
    "qwen3.5-9b":    "qwen3.5:9b",
    "gemma4-12b":    "gemma4:12b",
}
USE_DRIVE = True
WORKERS = 4        # questions answered at the same time
"""),
    md("## 2. Check the GPU"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "NO GPU: Runtime > Change runtime type > T4 GPU"
"""),
    md("## 3. Install and start Ollama"),
    code("""
!apt-get -qq install -y zstd > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh > /dev/null
import os, subprocess, time, requests
os.environ["OLLAMA_NUM_PARALLEL"] = "4"
os.environ["OLLAMA_KEEP_ALIVE"] = "30m"
server = subprocess.Popen(["ollama", "serve"], stdout=open("/content/ollama.log", "w"),
                          stderr=subprocess.STDOUT, env=os.environ.copy())
for _ in range(60):
    try:
        print("Ollama version:", requests.get("http://127.0.0.1:11434/api/version").json()["version"])
        break
    except requests.RequestException:
        time.sleep(1)
"""),
    md("## 4. Files"),
    code("""
import json
from google.colab import files
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    SAVE_DIR = "/content/drive/MyDrive/bayanbench"
else:
    SAVE_DIR = "/content/bayanbench"
os.makedirs(SAVE_DIR, exist_ok=True)

QUESTIONS_FILE = f"{SAVE_DIR}/questions_{SAMPLE}.jsonl"
if not os.path.exists(QUESTIONS_FILE):
    print(f"Please upload questions_{SAMPLE}.jsonl")
    up = files.upload()
    with open(QUESTIONS_FILE, "wb") as f:
        f.write(up[next(iter(up))])
questions = [json.loads(l) for l in open(QUESTIONS_FILE, encoding="utf-8")]
print(len(questions), "questions loaded")
"""),
    md("## 5. Client code (same prompt and settings for every model)"),
    code(CLIENT),
    md("## 6. Run every model"),
    code("""
summary = {}
for name, tag in MODELS.items():
    print(f"\\n=== {name} ({tag}) ===")
    t0 = time.time()
    pull(tag)
    print(f"downloaded in {time.time() - t0:.0f}s")
    results_file = f"{SAVE_DIR}/results_{name}_{SAMPLE}.jsonl"
    info = run_model(name, tag, questions, results_file, workers=WORKERS)
    summary[name] = dict(summarize(results_file, len(questions)), quantization=info["quantization"])
    unload(tag)

import pandas as pd
pd.DataFrame(summary).T
"""),
    md("""
## 7. Download the results
"cut_off" should be 0. If a model shows many unreadable answers or cut-offs, mention it when you upload.
"""),
    code("""
for name in MODELS:
    p = f"{SAVE_DIR}/results_{name}_{SAMPLE}.jsonl"
    if os.path.exists(p):
        files.download(p)
"""),
]


def main():
    nb = {"cells": CELLS, "metadata": {"colab": {"provenance": [], "gpuType": "T4"},
          "accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    out = ROOT / "notebooks" / "02_ask_open_models.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "notebooks" / "open_client.py").write_text(CLIENT.lstrip(), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
