"""Write notebooks/03_ask_gemini_batched.ipynb from notebooks/gemini_batch_client.py."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIENT = (ROOT / "notebooks" / "gemini_batch_client.py").read_text(encoding="utf-8")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


CELLS = [
    md("""
# BayanBench: Ask Gemini (batched)

The free Gemini tier allows only about 20 requests per day, so each request carries **25 questions**. One question type per request, and never two questions about the same town in one request, so no question reveals another's answer.

Run all cells once a day. It stops by itself at the daily limit and continues from there the next day.
"""),
    code("!pip install -q google-genai"),
    md("## 1. Settings"),
    code("""
MODEL = "gemini-3.6-flash"
SAMPLE = "pilot"        # first "pilot" (10 requests), then "core" (about 74 requests over several days)
BATCH_SIZE = 25
SECONDS_BETWEEN_REQUESTS = 15

import os, json
from google.colab import drive, files, userdata
drive.mount("/content/drive")
SAVE_DIR = "/content/drive/MyDrive/bayanbench"
os.makedirs(SAVE_DIR, exist_ok=True)
RESULTS_FILE = f"{SAVE_DIR}/results_{MODEL}-batch{BATCH_SIZE}_{SAMPLE}.jsonl"
QUESTIONS_FILE = f"{SAVE_DIR}/questions_{SAMPLE}.jsonl"
if not os.path.exists(QUESTIONS_FILE):
    print(f"Please upload questions_{SAMPLE}.jsonl")
    up = files.upload()
    open(QUESTIONS_FILE, "wb").write(up[next(iter(up))])
questions = [json.loads(l) for l in open(QUESTIONS_FILE, encoding="utf-8")]
print(len(questions), "questions; results go to", RESULTS_FILE)
"""),
    md("## 2. Batching code"),
    code(CLIENT),
    md("## 3. Connect to Gemini (no test request, to save the daily quota)"),
    code("""
from google import genai
from google.genai import types
try:
    API_KEY = userdata.get("GEMINI_API_KEY")
except Exception:
    API_KEY = userdata.get("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)
CONFIG = types.GenerateContentConfig(system_instruction=SYSTEM_BATCH, temperature=0.0,
                                     response_mime_type="application/json")
MODEL_VERSION = {}

def generate(prompt):
    resp = client.models.generate_content(model=MODEL, contents=prompt, config=CONFIG)
    MODEL_VERSION["v"] = getattr(resp, "model_version", None)
    return resp.text
"""),
    md("## 4. Run"),
    code("""
done = set()
if os.path.exists(RESULTS_FILE):
    for line in open(RESULTS_FILE, encoding="utf-8"):
        r = json.loads(line)
        if r.get("error") is None:
            done.add(r["qid"])

def write(rec):
    rec["model_version"] = MODEL_VERSION.get("v")
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\\n")

try:
    run_batches(generate, MODEL, questions, done, write, size=BATCH_SIZE,
                pause=SECONDS_BETWEEN_REQUESTS)
    ok = set()
    for line in open(RESULTS_FILE, encoding="utf-8"):
        r = json.loads(line)
        if r.get("error") is None:
            ok.add(r["qid"])
    left = len(questions) - len(ok)
    if left == 0:
        print("All questions in this sample are answered.")
    else:
        print(f"{left} questions still need an answer. Run this cell again to retry them.")
except DailyQuotaReached:
    print("Daily limit reached. Progress is saved. Run this notebook again tomorrow.")
"""),
    md("## 5. Progress and download"),
    code("""
latest = {}
for line in open(RESULTS_FILE, encoding="utf-8"):
    r = json.loads(line)
    if r.get("error") is None:
        latest[r["qid"]] = r
print(f"{len(latest)} of {len(questions)} questions answered")
files.download(RESULTS_FILE)
"""),
]

nb = {"cells": CELLS, "metadata": {"colab": {"provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "notebooks" / "03_ask_gemini_batched.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Wrote", out)
