"""A fake Ollama server for testing the notebook client without a GPU.

Mimics /api/pull, /api/show, /api/chat and /api/generate. Two fake models:
    plain    no thinking; answers in JSON; about 1 in 40 requests fails with HTTP 500
    thinker  has thinking that can be disabled; if 'think' is not false it puts
             a <think> block before the JSON; one fixed question gets a prose reply
Every /api/chat request body is logged so the test can check what was sent.
"""
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = []
_lock = threading.Lock()
_rng = random.Random(0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        model = req.get("model", "")
        if self.path == "/api/pull":
            return self._send(200, {"status": "success"})
        if self.path == "/api/generate":
            return self._send(200, {"done": True})
        if self.path == "/api/show":
            info = {"digest": f"sha256:{model}", "details": {"quantization_level": "Q4_K_M",
                    "parameter_size": "8B", "family": "llama"}, "capabilities": ["completion"]}
            if model == "thinker":
                info["capabilities"].append("thinking")
                info["thinking"] = {"values": [True, False], "default": True}
            return self._send(200, info)
        if self.path == "/api/chat":
            with _lock:
                LOG.append(req)
                fail = model == "plain" and _rng.random() < 1 / 40
            if fail:
                return self._send(500, {"error": "simulated server error"})
            q = req["messages"][-1]["content"]
            if "barangays are in" in q:
                ans = 12
            elif "Name three" in q:
                ans = ["Poblacion", "San Jose", "Santa Cruz"]
            elif "city or a municipality" in q:
                ans = "municipality"
            else:
                ans = "unknown"
            content = json.dumps({"answer": ans})
            thinking = ""
            if model == "thinker":
                if req.get("think") is not False:
                    content = "<think>let me think</think>" + content
                    thinking = "let me think"
                if "Solsona" in q and "Name three" in q:
                    content = "I am not sure about this town."
            return self._send(200, {"message": {"role": "assistant", "content": content,
                                                "thinking": thinking},
                                    "done_reason": "stop", "eval_count": 9})
        self._send(404, {"error": "not found"})


def start(port=11555):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
