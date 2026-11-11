#!/usr/bin/env python3
"""Resident LLM server for the Match 4 container.

Loads Qwen3-4B-Instruct-2507 ONCE (during the --index pass) and serves
newline-delimited JSON requests over a unix socket, so the ten per-question
processes never pay model-load time.

Backend selection:
    GZ_BACKEND=transformers (default) — bf16, explicit .to("cuda"), greedy
    GZ_BACKEND=stub — deterministic local heuristics for CPU tests; the stub
        deliberately keeps a VRAM-shaped sleep so timing paths are exercised.
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

SOCKET_PATH = os.environ.get("GZ_M4_SOCKET", "/tmp/gz_m4.sock")
MODEL_DIR = os.environ.get("GZ_MODEL_DIR", "/models/Qwen3-4B-Instruct-2507")
BACKEND = os.environ.get("GZ_BACKEND", "transformers")
MAX_NEW_TOKENS = int(os.environ.get("GZ_MAX_NEW_TOKENS", "64"))

_lock = threading.Lock()


# ----------------------------------------------------------------- stub

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[°]?c\b|°c|degrees? c|celsius", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


class StubBackend:
    """No-GPU backend for local development and unit tests."""

    name = "stub"

    def generate(self, query: str, pages: list[tuple[str, str]]) -> dict:
        time.sleep(0.05)
        qwords = [w for w in re.findall(r"[a-z0-9]+", query.lower())
                  if w not in ("what", "is", "the", "of", "in", "a", "to",
                               "and", "for", "on", "how", "many", "much")]
        best, best_hits, best_path = "", 0, ""
        for path, passage in pages:
            words = set(re.findall(r"[a-z0-9]+", passage.lower()))
            hits = sum(1 for w in qwords if w in words)
            if hits > best_hits:
                # crude value guess: nearest number to the query terms
                m = re.findall(r"\b\d[\d,.]*\b", passage)
                best, best_hits, best_path = (m[0] if m else ""), hits, path
        if not best:
            return {"answer": "", "citations": [], "confidence": 0.0}
        return {"answer": best, "citations": [best_path], "confidence": 0.4}


# ------------------------------------------------------------ transformers

class TransformersBackend:
    name = "transformers"

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = os.environ.get("GZ_DEVICE", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm device required for the real backend")
        model_path = MODEL_DIR
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype)
        model.to(self.device)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        # warm-up so the first graded request does not pay compile time
        ids = tokenizer("warm", return_tensors="pt").to(self.device)
        with torch.no_grad():
            model.generate(**ids, max_new_tokens=8, do_sample=False)
        if self.device == "cuda":
            torch.cuda.synchronize()

    def generate(self, query: str, pages: list[tuple[str, str]]) -> dict:
        numbered = "\n\n".join(
            f"[{i + 1}] path: {p}\n{t[:1500]}"
            for i, (p, t) in enumerate(pages))
        system = (
            "You extract one exact fact from retrieved web pages. "
            "Answer with the VALUE ONLY, exactly as it appears - no sentence, "
            "no explanation. If the pages do not contain the answer, answer "
            "with the single word EMPTY. Then on a new line list the page "
            "numbers that contain the answer, like: PAGES: 2"
        )
        user = f"Question: {query}\n\nRetrieved pages:\n{numbered}"
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(prompt, return_tensors="pt",
                             truncation=True, max_length=8192).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                temperature=None, top_p=None, top_k=None)
        text = self.tokenizer.decode(
            out[0][ids["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()

        parts = re.split(r"\bPAGES?\s*:", text, flags=re.IGNORECASE)
        answer = parts[0].strip().strip('"').strip()
        cited_nums = []
        if len(parts) > 1:
            cited_nums = [int(x) for x in re.findall(r"\d+", parts[1])[:4]]
        citations = []
        for num in cited_nums:
            if 1 <= num <= len(pages):
                p = pages[num - 1][0]
                if p not in citations:
                    citations.append(p)
        if not citations:
            # containment check: which retrieved pages hold the answer
            na = _norm(answer)
            for p, t in pages:
                if na and na in _norm(t):
                    citations.append(p)
                if len(citations) >= 2:
                    break
        if answer.upper().startswith("EMPTY") or not answer:
            return {"answer": "", "citations": [], "confidence": 0.0}
        return {"answer": answer, "citations": citations, "confidence": 0.85}


# ------------------------------------------------------------------ serve

def make_backend():
    if BACKEND == "stub":
        return StubBackend()
    try:
        return TransformersBackend()
    except Exception:
        traceback.print_exc()
        status = Path(SOCKET_PATH + ".status.json")
        status.write_text(json.dumps(
            {"backend": "transformers", "error": traceback.format_exc()[-2000:],
             "at": time.time()}))
        raise


def handle(conn, backend):
    with conn:
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(65536)
            if not chunk:
                return
            buf += chunk
        try:
            msg = json.loads(buf.split(b"\n", 1)[0].decode("utf-8", "replace"))
        except json.JSONDecodeError:
            conn.sendall(b'{"error":"bad json"}\n')
            return
        op = msg.get("op")
        if op == "ping":
            conn.sendall(json.dumps(
                {"ok": True, "backend": backend.name}).encode() + b"\n")
            return
        if op == "generate":
            with _lock:
                resp = backend.generate(msg.get("query", ""),
                                        msg.get("pages", []))
            resp.setdefault("answer", "")
            resp.setdefault("citations", [])
            conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode())
            return
        conn.sendall(b'{"error":"unknown op"}\n')


def main() -> int:
    backend = make_backend()
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET_PATH)
    srv.listen(16)
    Path(SOCKET_PATH + ".status.json").write_text(json.dumps(
        {"backend": backend.name, "ready": True, "at": time.time()}))
    print(f"[server] ready backend={backend.name} socket={SOCKET_PATH}",
          flush=True)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn, backend),
                         daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
