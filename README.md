# Ghostfox Zero — Match 4: Intelligent Web Information Retrieval

AMD AI League entry container. An agent that navigates a saved copy of a website,
understands a natural-language request, and retrieves the exact value the user needs —
with the source page as citation. Never a summary, never a guess.

## Architecture

- **Page walker + BM25 index** (`app/retrieval.py`): parses every HTML page in the corpus,
  follows all navigation links, cleans text, builds a full BM25 index. Zero downloads at eval time.
- **Resident LLM server** (`app/server.py`): loads Qwen3-4B-Instruct-2507 once in bf16 on the
  AMD GPU via ROCm PyTorch, serves over a unix socket, warm-ready within the startup budget.
- **Query client** (`app/app.py`): thin client — phase 1 `--index <corpus>` builds the index and
  starts the server; phase 2 `--corpus <corpus> --query-id N --query "..."` ranks passages with
  BM25, asks the model to extract the exact value, and writes `output/<qid>_output.json`.

## Output contract

```json
{"answer": "<value only>", "citations": ["path/to/page.html"], "confidence": 0.9}
```

Fail-closed: if the value cannot be verified from the corpus, the agent returns an empty
answer instead of guessing.

## Container

The image is built on the mandated ROCm PyTorch base (digest-pinned), with the model
weights baked in — fully offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).

```bash
docker pull ghcr.io/autokeren/ghostfox-zero-rocm:match4
```

Build script (crane layer-append, no Docker daemon): `tools/build.sh`

## Stack

AMD ROCm · PyTorch · Qwen3-4B-Instruct · BM25 retrieval · pure Python

---
Built solo by Ajat Sudrajat (AutoKeren) for the AMD AI League.
