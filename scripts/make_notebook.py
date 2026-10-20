"""Write notebooks/01_ask_gemini.ipynb (kept as a script so the notebook is reproducible)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


CELLS = [
    md("""
# BayanBench: Ask Gemini

Sends BayanBench questions to a Gemini model and saves the **raw answers**. Scoring happens separately, so this notebook never sees the answer key.

**Steps**
1. Add your Gemini key in Colab Secrets (key icon on the left) as `GEMINI_API_KEY`, with notebook access turned on.
2. Run the cells in order. When asked, upload `questions_pilot.jsonl` (or `questions_core.jsonl`).
3. Download the results file at the end.

Progress is saved after every answer. If the free daily limit is reached, run the notebook again later; it skips questions already answered.
"""),
    code("""
!pip install -q google-genai
"""),
    md("""
## 1. Settings
Saving to Google Drive keeps progress even if Colab disconnects. Set `USE_DRIVE = False` to save in this Colab session only.
"""),
    code("""
MODEL = "gemini-3.6-flash"      # change here if Google retires this model name
SAMPLE = "pilot"                 # "pilot" (232 questions) or "core" (1,825 questions)
SECONDS_BETWEEN_CALLS = 4.0      # raise this if you see many rate-limit pauses
USE_DRIVE = True

import os
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    SAVE_DIR = "/content/drive/MyDrive/bayanbench"
else:
    SAVE_DIR = "/content/bayanbench"
os.makedirs(SAVE_DIR, exist_ok=True)
RESULTS_FILE = f"{SAVE_DIR}/results_{MODEL}_{SAMPLE}.jsonl"
print("Results will be saved to:", RESULTS_FILE)
"""),
    md("## 2. Connect to Gemini"),
    code("""
from google import genai
from google.genai import types
from google.colab import userdata

try:
    API_KEY = userdata.get("GEMINI_API_KEY")
except Exception:
    API_KEY = userdata.get("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

test = client.models.generate_content(model=MODEL, contents="Reply with the single word: ready")
print("Model reply:", test.text.strip())
print("Exact model version:", getattr(test, "model_version", "not reported"))
"""),
    md("## 3. Load the questions"),
    code("""
import json
from google.colab import files

QUESTIONS_FILE = f"{SAVE_DIR}/questions_{SAMPLE}.jsonl"
if not os.path.exists(QUESTIONS_FILE):
    print(f"Please upload questions_{SAMPLE}.jsonl")
    uploaded = files.upload()
    name = next(iter(uploaded))
    with open(QUESTIONS_FILE, "wb") as f:
        f.write(uploaded[name])

questions = [json.loads(line) for line in open(QUESTIONS_FILE, encoding="utf-8")]
done = set()
if os.path.exists(RESULTS_FILE):
    for line in open(RESULTS_FILE, encoding="utf-8"):
        r = json.loads(line)
        if r.get("error") is None:
            done.add(r["qid"])
todo = [q for q in questions if q["qid"] not in done]
print(f"{len(questions)} questions, {len(done)} already answered, {len(todo)} to go")
print("Example:", questions[0]["question"])
"""),
    md("""
## 4. The prompt
The same instruction is used for every model in the benchmark. The model may answer "unknown"; honest uncertainty is scored separately from wrong answers.
"""),
    code("""
SYSTEM = '''You answer factual questions about places in the Philippines.
Reply with JSON only, in this format: {"answer": <your answer>}
- For a province or a region, give its name as a string.
- For "a city or a municipality", answer "city" or "municipality".
- For a number of barangays, give a whole number.
- For "name three barangays", give a list of three barangay names.
If you are not sure, set the answer to "unknown".'''

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    temperature=0.0,
    response_mime_type="application/json",
)

import re, time, datetime

def parse(text):
    if text is None:
        return None, "empty response"
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\\{.*\\}", t, re.S)
        if not m:
            return None, "not JSON"
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, "not JSON"
    if isinstance(obj, dict) and "answer" in obj:
        return obj["answer"], None
    return None, "no 'answer' field"

class QuotaReached(Exception):
    pass

def ask(question, max_retries=4):
    wait = 30
    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(model=MODEL, contents=question, config=CONFIG)
            return resp
        except Exception as e:
            msg = str(e)
            rate_limited = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if attempt == max_retries:
                if rate_limited:
                    raise QuotaReached(msg)
                raise
            print(f"  pause {wait}s ({'rate limit' if rate_limited else msg[:80]})")
            time.sleep(wait)
            wait *= 2
"""),
    md("## 5. Run"),
    code("""
answered = 0
try:
    for i, q in enumerate(todo, 1):
        record = {"qid": q["qid"], "qtype": q["qtype"], "model": MODEL,
                  "time": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            resp = ask(q["question"])
            text = resp.text
            answer, parse_error = parse(text)
            record.update(raw=text, answer=answer, parse_error=parse_error, error=None,
                          model_version=getattr(resp, "model_version", None))
        except QuotaReached as e:
            print("Daily limit reached. Progress is saved; run this notebook again later.")
            break
        except Exception as e:
            record.update(raw=None, answer=None, parse_error=None, error=str(e)[:300])
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\\n")
        answered += 1
        if i % 10 == 0 or i == len(todo):
            print(f"{i}/{len(todo)}  last: {q['question'][:60]} -> {record.get('answer')}")
        time.sleep(SECONDS_BETWEEN_CALLS)
finally:
    print(f"Saved {answered} new answers to {RESULTS_FILE}")
"""),
    md("## 6. Check and download"),
    code("""
import collections
rows = [json.loads(l) for l in open(RESULTS_FILE, encoding="utf-8")]
latest = {r["qid"]: r for r in rows}
print(f"{len(latest)} of {len(questions)} questions have a saved answer")
print("API errors:", sum(r["error"] is not None for r in latest.values()))
print("Unreadable replies:", sum(r.get("parse_error") is not None for r in latest.values()))
print("Answered 'unknown':", sum(str(r.get("answer")).strip().lower() == "unknown" for r in latest.values()))
print(collections.Counter(r["qtype"] for r in latest.values()))
files.download(RESULTS_FILE)
"""),
]


def main():
    nb = {"cells": CELLS, "metadata": {"colab": {"provenance": []},
          "kernelspec": {"name": "python3", "display_name": "Python 3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    out = ROOT / "notebooks" / "01_ask_gemini.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
