#!/usr/bin/env python3
"""Ghostfox Zero — Match 4 entry container (Intelligent Web Information Retrieval).

Implements the AMD AI League container contract:

  Phase 1 (startup budget, ungraded):
      python3 /app/app.py --index <corpus>
      Builds the page index and starts the resident LLM server, waits until
      ready, then exits 0.

  Phase 2 (30s per question, ten times):
      python3 /app/app.py --corpus <corpus> --query-id query_01 --query "..."
      Connects to the resident server, retrieves, extracts the exact value
      and writes /app/output/query_01_output.json

Output shape (missing keys are malformed and score zero):
      {"answer": "<value only>", "citations": ["path/to/page.html"], "confidence": 0.9}

The corpus for Match 4 is a saved copy of a website: HTML pages plus text
assets. The agent navigates it like a browser would — every page is parsed,
all navigation links are followed, and nothing is downloaded at evaluation
time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
SOCKET_PATH = os.environ.get("GZ_M4_SOCKET", "/tmp/gz_m4.sock")
SERVER_READY_TIMEOUT_S = float(os.environ.get("GZ_SERVER_TIMEOUT", "480"))

sys.path.insert(0, str(Path(__file__).parent))

from retrieval import build_index, load_index, retrieve_pages  # noqa: E402


# ---------------------------------------------------------------- arg parsing

QUESTION_FLAGS = ["--query", "--question", "--task", "--prompt", "-q", "--q"]
ID_FLAGS = ["--query-id", "--query_id", "--id", "--qid", "--task-id", "--task_id"]
START_FLAGS = ["--url", "--start-url", "--start_url", "--website", "--site",
               "--base-url", "--base_url"]
CORPUS_FLAGS = ["--corpus", "--sites", "--pages", "--web-root", "--web_root",
                "--root", "--sites-dir"]


def parse_args(argv):
    """Parse the contract flags plus every alias spelling the harness might use.

    Unknown long flags and their value are ignored, matching the lenient
    contract style of the earlier matches. A single positional JSON object is
    also accepted.
    """
    positional: list[str] = []
    opts: dict[str, str | None] = {}

    def set_once(key, value):
        opts.setdefault(key, value)

    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        if a == "--index":
            set_once("index", argv[i + 1] if i + 1 < n else "")
            i += 2 if i + 1 < n else 1
        elif a in QUESTION_FLAGS:
            val = argv[i + 1] if i + 1 < n else ""
            set_once("query", val)
            opts["_positional_suppressed"] = "1"
            i += 2 if i + 1 < n else 1
        elif a in ID_FLAGS:
            set_once("query_id", argv[i + 1] if i + 1 < n else "")
            i += 2 if i + 1 < n else 1
        elif a in START_FLAGS:
            set_once("url", argv[i + 1] if i + 1 < n else "")
            i += 2 if i + 1 < n else 1
        elif a in CORPUS_FLAGS:
            set_once("corpus", argv[i + 1] if i + 1 < n else "")
            i += 2 if i + 1 < n else 1
        elif a == "--input":
            set_once("input", argv[i + 1] if i + 1 < n else "")
            i += 2 if i + 1 < n else 1
        elif a.startswith("-"):
            # unknown flag: skip it and its directly-following value
            i += 2 if i + 1 < n and not argv[i + 1].startswith("-") else 1
        else:
            positional.append(a)
            i += 1

    if positional and "query" not in opts:
        p0 = positional[0]
        if p0.strip().startswith("{"):
            try:
                obj = json.loads(p0)
                for k, v in obj.items():
                    key = k.lstrip("-").replace("-", "_").replace(" ", "_")
                    if key in ("query", "question", "task", "prompt"):
                        set_once("query", str(v))
                    elif key in ("query_id", "id", "qid", "task_id"):
                        set_once("query_id", str(v))
                    elif key in ("url", "website", "site", "base_url"):
                        set_once("url", str(v))
                    elif key in ("corpus", "sites", "pages", "web_root", "root"):
                        set_once("corpus", str(v))
                opts["_positional_suppressed"] = "1"
            except json.JSONDecodeError:
                set_once("query", p0)
        else:
            set_once("query", p0)

    opts["_positional"] = [p for p in positional if not p.strip().startswith("{")]
    return opts


def sanitize_id(qid: str | None) -> str:
    """The output stem is the id the harness assigns, made filesystem-safe."""
    if not qid:
        return "query"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", qid)


# ------------------------------------------------------------------- sockets

def rpc(payload: dict, timeout_s: float = 25.0) -> dict:
    """One newline-delimited JSON request to the resident server."""
    import socket
    deadline = time.monotonic() + timeout_s
    data = (json.dumps(payload) + "\n").encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(max(1.0, timeout_s))
        s.connect(SOCKET_PATH)
        s.sendall(data)
        chunks = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("server deadline exceeded")
            s.settimeout(min(remaining, 5.0))
            chunk = s.recv(65536)
            if not chunk:
                break
            if b"\n" in chunk:
                chunks.append(chunk.split(b"\n", 1)[0])
                break
            chunks.append(chunk)
    raw = b"".join(chunks).decode("utf-8", "replace").strip()
    return json.loads(raw) if raw else {}


def wait_server_ready(timeout_s: float = SERVER_READY_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not os.path.exists(SOCKET_PATH):
            time.sleep(2)
            continue
        try:
            resp = rpc({"op": "ping"}, timeout_s=5.0)
            if resp.get("ok"):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def ensure_server_running() -> bool:
    """Make sure the resident LLM server is up; start it if it is not."""
    if os.path.exists(SOCKET_PATH):
        try:
            if rpc({"op": "ping"}, timeout_s=5.0).get("ok"):
                return True
        except Exception:
            pass
    import subprocess
    here = Path(__file__).parent
    subprocess.Popen(
        [sys.executable, str(here / "server.py")],
        stdout=open(os.environ.get("GZ_SERVER_LOG", "/tmp/gz_m4_server.log"), "ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return wait_server_ready()


# -------------------------------------------------------------------- phases

def phase_index(corpus: Path) -> int:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    stats = build_index(corpus, INDEX_DIR)
    print(f"[index] {stats['pages']} pages indexed, {stats['skipped']} skipped",
          flush=True)
    ok = ensure_server_running()
    if not ok:
        print("[index] WARNING: server not ready before returning", flush=True)
    return 0


def phase_query(corpus: Path, query: str, query_id: str) -> int:
    index = load_index(INDEX_DIR, corpus)
    pages = retrieve_pages(index, query, top_k=8)
    ensure_server_running()
    resp = rpc({"op": "generate", "query": query,
                "pages": [(p["path"], p["passage"]) for p in pages]},
               timeout_s=24.0)
    answer = (resp.get("answer") or "").strip()
    citations = list(resp.get("citations") or [])

    # Refuse rather than guess: a fabricated-looking answer helps nobody.
    if answer.lower() in ("", "n/a", "none", "unknown", "not found", "not present",
                          "not in pages", "no answer", "empty"):
        answer, citations = "", []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (sanitize_id(query_id) + "_output.json")
    out.write_text(
        json.dumps({"answer": answer, "citations": citations,
                    "confidence": float(resp.get("confidence") or 0.0)},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


def phase_batch(input_path: Path, corpus: Path | None) -> int:
    raw = input_path.read_text(encoding="utf-8", errors="replace").strip()
    items = []
    if raw.startswith("["):
        items = json.loads(raw)
    else:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
    for it in items:
        qid = sanitize_id(it.get("query_id") or it.get("id"))
        q = it.get("query") or it.get("question") or ""
        c = Path(it.get("corpus")) if it.get("corpus") else corpus
        if not c or not q:
            continue
        phase_query(c, q, qid)
    return 0


def main() -> int:
    opts = parse_args(sys.argv[1:])

    if opts.get("input"):
        return phase_batch(Path(opts["input"]),
                           Path(opts["corpus"]) if opts.get("corpus") else None)

    if opts.get("index"):
        target = Path(opts["index"])
        if "corpus" not in opts and target.is_dir():
            pass
        elif not opts.get("corpus"):
            opts["corpus"] = str(target)
        return phase_index(Path(opts.get("corpus") or opts["index"]))

    if opts.get("index") is None and (opts.get("query") or opts.get("corpus")):
        query = opts.get("query") or ""
        qid = opts.get("query_id") or "query"
        corpus = Path(opts["corpus"]) if opts.get("corpus") else None
        if opts.get("url") and (corpus is None or not corpus.exists()):
            corpus = Path("/app/corpus")
            if not corpus.exists():
                corpus = Path(opts["url"].replace("file://", ""))
        if corpus is None or not query:
            print("usage: app.py --corpus DIR --query-id ID --query Q", file=sys.stderr)
            return 2
        return phase_query(corpus, query, qid)

    print("usage: app.py --index DIR | --corpus DIR --query-id ID --query Q",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
