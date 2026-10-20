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
