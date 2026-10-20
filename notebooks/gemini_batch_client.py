"""Batched question answering for Gemini (used by notebooks/03_ask_gemini_batched.ipynb).

The free tier allows very few requests per day, so each request carries a batch
of questions. Rules that keep batching from leaking answers:
  * a batch holds ONE question type only
  * a batch never holds two questions about the same town
Questions about a town's province or region never show its province, while
other question types do; keeping types and towns apart prevents one question
from revealing another's answer.
"""
import json
import random
import re
import time
import datetime

SYSTEM = """You answer factual questions about places in the Philippines.
Reply with JSON only, in this format: {"answer": <your answer>}
- For a province or a region, give its name as a string.
- For "a city or a municipality", answer "city" or "municipality".
- For a number of barangays, give a whole number.
- For "name three barangays", give a list of three barangay names.
If you are not sure, set the answer to "unknown"."""

# Same rules as the single-question prompt used for every other model;
# only the reply format differs (a list of answers with ids).
SYSTEM_BATCH = """You answer factual questions about places in the Philippines.
You will receive several numbered questions. Answer each one independently.
- For a province or a region, give its name as a string.
- For "a city or a municipality", answer "city" or "municipality".
- For a number of barangays, give a whole number.
- For "name three barangays", give a list of three barangay names.
If you are not sure, set the answer to "unknown".
Reply with JSON only, in this format:
{"answers": [{"id": "<question id>", "answer": <your answer>}, ...]}
Include every id exactly once."""


def make_batches(questions, size=25, seed=2026):
    """Deterministic batches: one question type per batch, no repeated town."""
    by_type = {}
    for q in questions:
        by_type.setdefault(q["qtype"], []).append(q)
    batches = []
    for qtype in sorted(by_type):
        qs = sorted(by_type[qtype], key=lambda q: q["qid"])
        random.Random(f"{seed}-{qtype}").shuffle(qs)
        for i in range(0, len(qs), size):
            batches.append(qs[i:i + size])
    for b in batches:
        assert len({q["lgu_code"] for q in b}) == len(b), "town repeated in a batch"
        assert len({q["qtype"] for q in b}) == 1, "mixed question types in a batch"
    return batches


def batch_prompt(batch):
    lines = [f'{q["qid"]}: {q["question"]}' for q in batch]
    return "Questions:\n" + "\n".join(lines)


def parse_batch(text):
    """Return {qid: answer} and an error message (None if fine)."""
    if not text:
        return {}, "empty response"
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return {}, "not JSON"
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}, "not JSON"
    items = obj.get("answers") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        return {}, "no 'answers' list"
    out = {}
    for it in items:
        if isinstance(it, dict) and "id" in it and "answer" in it:
            out[str(it["id"]).strip()] = it["answer"]
    return out, None


class DailyQuotaReached(Exception):
    pass


def classify_error(msg):
    if "PerDay" in msg or "per day" in msg.lower():
        return "daily"
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "rate"
    if "503" in msg or "UNAVAILABLE" in msg or "500" in msg:
        return "busy"
    return "other"


def run_batches(generate, model_name, questions, done, write, size=25,
                pause=15, max_requests=None, sleep=time.sleep, log=print):
    """generate(prompt_text) -> response text. write(record) saves one answer record.

    Returns the number of requests sent. Stops cleanly at the daily quota.
    """
    todo = [q for q in questions if q["qid"] not in done]
    batches = make_batches(todo, size)
    log(f"{len(todo)} questions to go in {len(batches)} requests")
    sent = 0
    for bi, batch in enumerate(batches, 1):
        if max_requests is not None and sent >= max_requests:
            log("Reached the request limit set for this run.")
            break
        text, err, wait = None, None, 30
        for attempt in range(3):
            try:
                sent += 1
                text = generate(batch_prompt(batch))
                err = None
                break
            except Exception as e:  # noqa: BLE001
                kind = classify_error(str(e))
                if kind == "daily":
                    raise DailyQuotaReached(str(e)[:200])
                err = str(e)[:300]
                if kind in ("rate", "busy") and attempt < 2:
                    log(f"  pause {wait}s ({kind})")
                    sleep(wait)
                    wait *= 2
                    continue
                break
        answers, perr = parse_batch(text) if err is None else ({}, None)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        got = 0
        for pos, q in enumerate(batch):
            rec = {"qid": q["qid"], "qtype": q["qtype"], "model": model_name, "mode": f"batch{size}",
                   "batch_index": bi, "batch_position": pos, "time": now}
            if err is not None:
                rec.update(raw=None, answer=None, parse_error=None, error=err)
            elif perr is not None:  # whole reply unreadable: a format failure, so retry next run
                rec.update(raw=text, answer=None, parse_error=None, error=f"unreadable batch reply: {perr}")
            elif q["qid"] in answers:
                rec.update(raw=None, answer=answers[q["qid"]], parse_error=None, error=None)
                got += 1
            else:  # model skipped this id: retry next run
                rec.update(raw=None, answer=None, parse_error=None, error="missing from batch reply")
            write(rec)
        reason = ""
        if err is not None:
            reason = f"  (failed: {classify_error(err)}; will retry next run)"
        elif perr is not None:
            reason = f"  (reply unreadable: {perr}; will retry next run)"
        elif got < len(batch):
            reason = "  (some ids missing; will retry next run)"
        log(f"request {bi}/{len(batches)} ({batch[0]['qtype']}): {got}/{len(batch)} answers{reason}")
        sleep(pause)
    return sent
