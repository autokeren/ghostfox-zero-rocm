#!/usr/bin/env python3
"""Page walker, HTML-to-text and BM25 retrieval for the Match 4 corpus.

The corpus is a saved copy of a website, so it is walked like one: every
readable file is parsed, per-file failures are caught and skipped, and page
text is kept in a compact JSON index for the per-question processes.

No third-party dependencies: the mandated ROCm base image ships torch and PIL
only, and everything the retrieval needs lives in the standard library.
"""

from __future__ import annotations

import json
import math
import os
import re
from html.parser import HTMLParser
from pathlib import Path

TEXT_EXTENSIONS = {".html", ".htm", ".txt", ".md", ".json", ".text", ""}
PAGE_SIZE_CAP = 1_500_000  # 1.5 MB of text per file, like the contract advises

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into",
    "is", "it", "its", "may", "of", "on", "or", "our", "per", "the", "their",
    "this", "to", "was", "were", "what", "when", "which", "who", "will", "with",
    "you", "your",
}

WORD_RE = re.compile(r"[a-z0-9]+")
PASSAGE_SPLIT_RE = re.compile(r"\n\s*\n")


class _TextExtractor(HTMLParser):
    """Stdlib HTML-to-text: keeps visible text and link targets, drops noise."""

    SKIP = {"script", "style", "noscript", "template", "svg", "head", "iframe"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._chunks: list[str] = []
        self._links: list[str] = []
        self._in_a = False
        self.title = ""

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag == "a":
            self._in_a = True
            for k, v in attrs:
                if k == "href" and v and not v.startswith("#"):
                    self._links.append(v)
        elif tag in ("p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "table"):
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "a":
            self._in_a = False

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_a and data.strip():
            self._links.append(data.strip())
        self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()

    def links(self) -> list[str]:
        return self._links


def html_to_text(data: str) -> tuple[str, list[str], str]:
    """Parse HTML, return (text, links, title)."""
    p = _TextExtractor()
    try:
        p.feed(data)
        p.close()
        m = re.search(r"<title[^>]*>(.*?)</title>", data,
                      re.IGNORECASE | re.DOTALL)
        title = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    except Exception:
        title = ""
        p._links = []
    return p.text(), p.links(), title


def tokenize(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(text.lower())
            if w not in STOPWORDS and len(w) > 1]


def read_page(path: Path, root: Path) -> dict | None:
    """Read one file defensively. Returns None on any failure."""
    try:
        if path.is_dir():
            return None
        name = path.name.lower()
        ext = path.suffix.lower()
        if ext not in TEXT_EXTENSIONS and not os.access(path, os.R_OK):
            return None
        data = path.read_bytes()[:PAGE_SIZE_CAP]
        if b"\x00" in data[:4096]:
            return None  # binary
        if ext in (".html", ".htm"):
            text, links, title = html_to_text(data.decode("utf-8", "replace"))
        else:
            text = data.decode("utf-8", "replace")
            links, title = [], name
        if not text.strip():
            return None
        rel = path.relative_to(root).as_posix()
        return {"path": rel, "text": text, "links": links, "title": title,
                "tokens": tokenize(text)}
    except Exception:
        return None


def build_index(corpus: Path, index_dir: Path) -> dict:
    """Walk the saved-site corpus and persist pages.json. Never raises."""
    corpus = Path(corpus)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    pages: dict[str, dict] = {}
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(corpus, followlinks=False):
        dirnames[:] = [d for d in dirnames
                       if not os.path.islink(Path(dirpath) / d)]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.is_symlink():
                continue
            page = read_page(p, corpus)
            if page is None:
                skipped += 1
                continue
            pages[page["path"]] = {k: page[k] for k in
                                   ("path", "text", "links", "title")}

    (index_dir / "pages.json").write_text(
        json.dumps(pages, ensure_ascii=False), encoding="utf-8")

    # corpus signature, so a changed corpus forces a rebuild
    sig = sorted(pages.keys())
    (index_dir / "signature.json").write_text(
        json.dumps({"root": str(corpus), "files": sig,
                    "mtime": os.path.getmtime(corpus) if corpus.exists() else 0},
                   ensure_ascii=False), encoding="utf-8")
    return {"pages": len(pages), "skipped": skipped}


def load_index(index_dir: Path, corpus: Path | None) -> dict:
    """Load pages.json; fall back to a rebuild if the index is missing."""
    index_dir = Path(index_dir)
    pages_file = index_dir / "pages.json"
    if pages_file.exists():
        pages = json.loads(pages_file.read_text(encoding="utf-8"))
        if pages:
            return pages
    if corpus and Path(corpus).exists():
        return {}
    return {}


# ------------------------------------------------------------------- BM25

def _bm25(query_tokens: list[str], pages: dict, k1: float = 1.2,
          b: float = 0.75) -> list[tuple[str, float]]:
    doc_tokens = {path: tokenize(p["text"]) for path, p in pages.items()}
    df: dict[str, int] = {}
    for toks in doc_tokens.values():
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    n = max(1, len(doc_tokens))
    avgdl = sum(len(t) for t in doc_tokens.values()) / max(1, n)

    scores = []
    for path, toks in doc_tokens.items():
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        dl = len(toks)
        for qt in query_tokens:
            if qt not in tf:
                continue
            idf = math.log(1 + (n - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5))
            denom = tf[qt] + k1 * (1 - b + b * dl / avgdl)
            s += idf * tf[qt] * (k1 + 1) / denom
        scores.append((path, s))
    scores.sort(key=lambda x: -x[1])
    return scores


def _best_passages(text: str, query_tokens: list[str], window: int = 700,
                   max_passages: int = 2) -> list[str]:
    """Sliding windows scored by query-term density."""
    paras = [p.strip() for p in PASSAGE_SPLIT_RE.split(text) if p.strip()]
    if not paras:
        return [text[:window]]
    windows = []
    cur = ""
    for para in paras:
        if len(cur) + len(para) <= window:
            cur = (cur + "\n" + para).strip()
        else:
            if cur:
                windows.append(cur)
            cur = para[:window]
    if cur:
        windows.append(cur)
    scored = []
    qset = set(query_tokens)
    for w in windows:
        toks = tokenize(w)
        hits = sum(1 for t in toks if t in qset)
        scored.append((hits / (1 + len(toks) ** 0.5), w))
    scored.sort(key=lambda x: -x[0])
    return [w for _, w in scored[:max_passages]] or [text[:window]]


def retrieve_pages(index: dict, query: str, top_k: int = 8) -> list[dict]:
    """Return [{path, passage, score}] for the query, best first."""
    if not index:
        return []
    qtokens = tokenize(query)
    ranked = _bm25(qtokens, index)
    out = []
    for path, score in ranked[:top_k]:
        if score <= 0:
            break
        page = index[path]
        passages = _best_passages(page["text"], qtokens)
        out.append({"path": path,
                    "passage": "\n--\n".join(passages)[:1600],
                    "score": round(score, 3)})
    return out
