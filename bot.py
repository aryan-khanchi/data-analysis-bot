"""
Data Analyst Telegram Bot
-------------------------
One process that does three things:
1. Serves a tiny website (so your run logs have a public URL)
2. Polls Telegram for new messages
3. Runs an LLM agent that can search the web, download files and run Python

Run locally: python bot.py
"""
import base64
import os
import re
import hashlib
import inspect
import io
import json
import sys
import time
import uuid
import threading
import queue
import subprocess
import traceback
from pathlib import Path

import requests
from flask import Flask, send_from_directory, jsonify
from openai import OpenAI

try:  # loads your .env file when running locally
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --------------------------------------------------------------------------
# Configuration (all of this comes from your .env file)
# --------------------------------------------------------------------------
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("MODEL", "gpt-4.1-mini")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
PORT = int(os.environ.get("PORT", 8000))

# Durable log storage (optional but strongly recommended - Render wipes its disk)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO")  # e.g. "avi/tds-telegram-bot"
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

# Memory guards - the free tier gives us 512MB for everything.
# MAX_DOWNLOAD_MB should be set to 60 in your .env - real government PDFs
# (PLFS, MOSPI, RBI reports) are routinely 20-50MB, and the old 8MB default
# rejected them outright before the agent ever got to read them.
MAX_DOWNLOAD = int(os.environ.get("MAX_DOWNLOAD_MB", 60)) * 1_000_000
CHILD_MEM_MB = int(os.environ.get("CHILD_MEM_MB", 300))

TG_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Reasoning / "thinking" tokens. Empty = off (the default for lite-tier models).
# Set REASONING_EFFORT to low / medium / high to make the model deliberate before
# acting. OpenRouter normalises this across providers. NOTE: reasoning tokens are
# billed as OUTPUT tokens and add latency, so they eat both your credit and your
# TIME_BUDGET - start with "low".
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "").strip().lower()

MAX_STEPS = int(os.environ.get("MAX_STEPS", 34))
# Raised because runs now finish well inside TIME_BUDGET and hit this limit
# instead: a run that used only 100s of 240s still exhausted 24 steps. Every
# tool call is clamped to the remaining time (see _time_left), so a bigger step
# budget cannot push the run past the deadline - the deadline still governs.
                                                   # (SEARCH_CALL_LIMIT), so most of this
                                                   # budget should go toward actually reading
                                                   # documents rather than re-searching
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", 240))
# Grounded in the real pipeline (Jivraj-18/tds-p1-t2-2026-telegram-bot):
# collect.py uses question.get("timeout_seconds", 300) and the README's own
# question template sets 300, so 300 is the working ceiling. 240 leaves 60s for
# the final forced submit_answer turn, the GitHub log publish and the Telegram
# send. Note timeout_seconds covers the WHOLE (possibly multi-turn) exchange,
# so a 3-message question shares one window across all three replies - another
# reason not to spend the full ceiling on any single reply.

# Seconds a single find_pdf_pages call may spend extracting text before it stops,
# saves progress, and returns what it has. Kept well under the subprocess timeout
# so a partial answer always gets back to the agent.
PDF_SCAN_BUDGET = int(os.environ.get("PDF_SCAN_BUDGET", 70))

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Remembers the last few messages per chat, for multi-turn questions
history: dict[int, list[str]] = {}
last_seen: dict[int, float] = {}
CONV_GAP = 240   # a gap this long means a NEW conversation started
STALE_AFTER = 300  # matches collect.py's default timeout_seconds (300). A message
                   # older than this was abandoned by the grader; replying now would
                   # land our answer inside the NEXT question's conversation.


OFFICIAL_DOMAINS = ("mospi.gov.in", "pib.gov.in", "data.gov.in", "sansad.in",
                    "rbi.org.in", "censusindia.gov.in", "esankhyiki.mospi.gov.in",
                    "microdata.gov.in", "dge.gov.in")


def _extract_candidate_urls(text: str) -> list[str]:
    """Pull out links to official-domain documents from a block of search
    results, so we can point the agent back at what it already found instead
    of letting it search for the same thing over and over."""
    urls = re.findall(r"https?://[^\s)\]]+", text)
    seen, out = set(), []
    for u in urls:
        u = u.rstrip(".,")
        if any(d in u for d in OFFICIAL_DOMAINS) and u not in seen:
            seen.add(u)
            out.append(u)
    return out


# Per-chat-worker deadline, so a tool can't run past the agent's deadline.
# collect.py awaits each reply within timeout_seconds for the WHOLE exchange
# (default 300), and our deadline is only checked BETWEEN steps - so without this
# a run_python started 1s before the deadline would add another 120s and blow
# straight past the grader's ceiling. One worker thread per chat, hence threading.local.
_ctx = threading.local()


def _time_left(default: int = 120) -> int:
    """Seconds until this run's deadline. Never returns less than 5 so a tool
    still gets a chance to fail fast rather than being handed a zero timeout."""
    deadline = getattr(_ctx, "deadline", None)
    if deadline is None:
        return default
    return max(5, int(deadline - time.time()))


def llm(messages, log, step, deadline=None):
    """Call the model, retrying politely when we hit the rate limit."""
    last_error = None
    kwargs = {}
    if REASONING_EFFORT in ("low", "medium", "high"):
        # OpenRouter's unified reasoning parameter. Passed via extra_body because
        # it isn't part of the standard OpenAI schema the SDK knows about.
        kwargs["extra_body"] = {"reasoning": {"effort": REASONING_EFFORT}}
    for attempt in range(6):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOL_SCHEMA, temperature=0,
                **kwargs,
            )
        except Exception as e:
            last_error = e
            text = str(e)
            rate_limited = ("429" in text or "RESOURCE_EXHAUSTED" in text
                            or "rate limit" in text.lower())
            if not rate_limited:
                raise
            m = re.search(r"retry(?:Delay|.{0,12}in)\D{0,4}(\d+(?:\.\d+)?)\s*s", text)
            wait = float(m.group(1)) + 2 if m else min(60, 5 * 2 ** attempt)
            if deadline and time.time() + wait > deadline:
                log.write("rate_limited_gave_up", step=step,
                          needed=round(wait, 1), left=round(deadline - time.time(), 1))
                raise
            log.write("rate_limited", step=step, attempt=attempt, sleeping=round(wait, 1))
            time.sleep(wait)
    raise last_error


# --------------------------------------------------------------------------
# Run log: one JSON object per line, served publicly at /logs/<id>.jsonl
# --------------------------------------------------------------------------
class RunLog:
    def __init__(self):
        self.run_id = uuid.uuid4().hex[:12]
        self.path = LOG_DIR / f"{self.run_id}.jsonl"
        self.path.touch()

    def write(self, event: str, **fields):
        record = {"ts": time.time(), "run_id": self.run_id, "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @property
    def url(self):
        return f"{PUBLIC_BASE_URL}/logs/{self.run_id}.jsonl"

    @property
    def public_url(self):
        if GITHUB_TOKEN and GITHUB_REPO:
            return (f"https://raw.githubusercontent.com/{GITHUB_REPO}/"
                    f"{GITHUB_BRANCH}/runs/{self.run_id}.jsonl")
        return self.url

    def publish(self):
        """Mirror the finished log to GitHub. Render's disk is wiped on every
        restart, so a log served only from here dies with the container."""
        if not (GITHUB_TOKEN and GITHUB_REPO):
            return self.url
        try:
            payload = base64.b64encode(self.path.read_bytes()).decode()
            r = requests.put(
                f"https://api.github.com/repos/{GITHUB_REPO}/contents/runs/{self.run_id}.jsonl",
                headers={"Authorization": f"Bearer {GITHUB_TOKEN}",
                         "Accept": "application/vnd.github+json"},
                json={"message": f"run log {self.run_id}", "content": payload,
                      "branch": GITHUB_BRANCH},
                timeout=30,
            )
            if r.status_code in (200, 201):
                return (f"https://raw.githubusercontent.com/{GITHUB_REPO}/"
                        f"{GITHUB_BRANCH}/runs/{self.run_id}.jsonl")
            print(f"github publish failed {r.status_code}: {r.text[:200]}", flush=True)
        except Exception as e:
            print(f"github publish error: {e}", flush=True)
        return self.url


# --------------------------------------------------------------------------
# TOOLS - the actions the agent is allowed to take
# --------------------------------------------------------------------------
_SEARCH_STOPWORDS = {
    "what", "which", "where", "when", "with", "from", "that", "this", "have",
    "does", "the", "and", "for", "was", "were", "site", "filetype", "http",
    "https", "www", "com", "org", "pdf", "html",
}


def _strip_search_operators(q: str) -> str:
    """This search backend does not honour site:/filetype: and returns unrelated
    junk when they're used. The system prompt says not to use them; the model
    does anyway, so strip them here."""
    q = re.sub(r"\b(?:site|filetype|inurl|intitle)\s*:\s*\S+", " ", q, flags=re.I)
    q = re.sub(r"\bOR\b", " ", q)
    q = q.replace('"', " ")
    return re.sub(r"\s+", " ", q).strip()


def _query_terms(q: str) -> set:
    return {w for w in re.findall(r"[a-z0-9\-]{4,}", q.lower())
            if w not in _SEARCH_STOPWORDS}


def _distinctive_terms(q: str, limit: int = 6) -> list:
    """Pick the terms most worth keeping when simplifying a failed query.

    Ordering by LENGTH is wrong: it throws away exactly the terms that identify
    the subject. For 'Periodic Labour Force Survey Annual Report 2023-24' the
    longest words are periodic/labour/survey/report - all generic - while the
    single most identifying token, PLFS, is only 4 characters and gets dropped.
    So: acronyms first (they were capitalised in the original), then tokens
    carrying digits (years, table numbers), then everything else, each group in
    the order the model wrote them."""
    acronyms, numeric, rest, seen = [], [], [], set()
    for raw in re.findall(r"[A-Za-z0-9\-]{3,}", q):
        low = raw.lower()
        if low in _SEARCH_STOPWORDS or low in seen:
            continue
        seen.add(low)
        if raw.isupper() and raw.isalpha():
            acronyms.append(raw)
        elif any(c.isdigit() for c in raw):
            numeric.append(raw)
        elif len(low) >= 4:
            rest.append(raw)
    return (acronyms + numeric + rest)[:limit]


def _relevance(query: str, results: str) -> float:
    """Fraction of the query's distinctive terms that appear anywhere in the
    results. Near zero means the backend returned something unrelated -
    e.g. Dropbox blog posts for a PLFS query, which really happens."""
    terms = _query_terms(query)
    if not terms:
        return 1.0
    low = results.lower()
    return sum(1 for t in terms if t in low) / len(terms)


SEARCH_RELEVANCE_FLOOR = float(os.environ.get("SEARCH_RELEVANCE_FLOOR", 0.4))

# ddgs can query several independent engines. The default is backend="auto",
# which picks for us - and when its pick returns junk there is no second opinion.
# Naming engines explicitly lets us retry on a DIFFERENT one, which is far more
# effective than rewording a query for an engine that is having a bad day.
# Available (ddgs text): brave, duckduckgo, google, mojeek, startpage, yahoo,
# yandex, wikipedia, grokipedia.
SEARCH_BACKENDS = [b.strip() for b in os.environ.get(
    "SEARCH_BACKENDS", "duckduckgo,google,brave,mojeek").split(",") if b.strip()]

OFF_TOPIC_MARKER = "[OFF-TOPIC RESULTS]"


def _ddg(queries, backend=None) -> str:
    from ddgs import DDGS
    out = []
    kwargs = {"max_results": 8}
    if backend:
        kwargs["backend"] = backend
    with DDGS() as ddg:
        for q in queries[:3]:
            try:
                hits = list(ddg.text(str(q), **kwargs))
            except TypeError:
                # older/newer ddgs without the backend kwarg - don't lose the search
                hits = list(ddg.text(str(q), max_results=8))
            out.append(f"### results for: {q}")
            if hits:
                out.extend(f"- {h.get('title')}\n  {h.get('href')}\n  {h.get('body','')[:300]}"
                           for h in hits)
            else:
                out.append("(no results)")
    return "\n".join(out)


def tool_search_web(query) -> str:
    """Search the web. Accepts one query or a list of them.

    This backend intermittently returns results for a completely different
    subject: it finds nothing for a long quoted query, then falls back to
    matching one generic word from it. Real observed cases - 'Periodic Labour
    Force Survey' returned the periodic table, 'Annual Report PLFS' returned
    dictionary entries for 'annual', and a PLFS query returned Dropbox blog
    posts. So we score every result set against the query and, when it looks
    off-topic, retry on a different engine with a simplified query before
    handing anything back."""
    queries = query if isinstance(query, list) else [query]
    cleaned = [_strip_search_operators(str(q)) or str(q) for q in queries]
    joined = " ".join(cleaned)

    attempts = []
    try:
        for i, backend in enumerate([None] + SEARCH_BACKENDS):
            # Each backend is a fresh network round trip (2-5s). Don't keep
            # shopping for a good engine when the run is nearly out of time -
            # hand back what we have and let the agent decide.
            if i > 0 and _time_left() < 30:
                break
            # First pass: the query as written. Later passes: a different engine
            # and, from the second retry on, a simplified query too.
            if i == 0:
                qs, label = cleaned, "as written"
            else:
                terms = _distinctive_terms(joined)
                qs = [" ".join(terms)] if terms else cleaned
                label = f"backend={backend}, simplified to '{qs[0]}'"

            result = _ddg(qs, backend=backend)
            score = _relevance(joined, result)
            attempts.append((score, label, result))
            if score >= SEARCH_RELEVANCE_FLOOR:
                if i == 0:
                    return result
                return f"[first attempt was off-topic; retried: {label}]\n{result}"

        # Nothing cleared the bar - hand back the least-bad set, clearly flagged.
        score, label, result = max(attempts, key=lambda a: a[0])
        return (f"{OFF_TOPIC_MARKER} Tried {len(attempts)} engines; none returned "
                f"results matching your query (best relevance {score:.2f}). This is a "
                f"backend fault, not your wording. DO NOT read any fact out of the "
                f"text below. Either open a source URL you already have, or try "
                f"plain keywords with no quotes and no site: filters.\n"
                f"[best of a bad set - {label}]\n{result}")
    except Exception as e:
        return f"search failed: {e}"


DOWNLOAD_DIR = Path("/tmp/downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _cached_download(url: str):
    """Fetch a URL to disk once. Later calls for the same URL are free.
    Returns (path, from_cache) or raises.

    Also accepts a LOCAL PATH that was already downloaded earlier (e.g. the
    path returned by download_file). The agent is told to reuse files by
    path instead of re-downloading them - so every tool that calls this must
    handle being handed a path, not just a URL, or it fails instantly and the
    agent falls back to much slower ways of reading the file."""
    maybe_path = Path(url)
    if maybe_path.exists() and maybe_path.is_file():
        return maybe_path, True

    ext = ""
    for e in (".pdf", ".xlsx", ".xls", ".csv", ".json", ".zip"):
        if url.lower().split("?")[0].endswith(e):
            ext = e
            break
    path = DOWNLOAD_DIR / (hashlib.sha1(url.encode()).hexdigest()[:16] + ext)
    if path.exists() and path.stat().st_size > 0:
        return path, True
    r = requests.get(url, timeout=min(120, _time_left()),
                     headers={"User-Agent": "Mozilla/5.0"}, stream=True)
    r.raise_for_status()
    size = 0
    with path.open("wb") as f:  # straight to disk, never all in RAM
        for chunk in r.iter_content(65536):
            size += len(chunk)
            if size > MAX_DOWNLOAD:
                f.close()
                path.unlink(missing_ok=True)
                raise ValueError(f"file exceeds {MAX_DOWNLOAD // 1_000_000}MB")
            f.write(chunk)
    return path, False


# Page counts of PDFs we've downloaded, so tool_run_python can cheaply tell
# whether a snippet is about to loop over a huge document. Filled by download_file.
PDF_PAGE_COUNTS: dict[str, int] = {}
LARGE_PDF_PAGES = int(os.environ.get("LARGE_PDF_PAGES", 50))


def tool_download_file(url: str) -> str:
    """Download once, keep it on disk, and report what's inside. Safe to call
    again with the SAME url later - it's free, since the file is already on disk."""
    try:
        path, cached = _cached_download(url)
        size_mb = path.stat().st_size / 1e6
        note = "already downloaded earlier" if cached else "downloaded"
        info = [f"{note}: {path} ({size_mb:.1f} MB)"]
        if path.suffix == ".pdf":
            try:
                from pypdf import PdfReader
                n = len(PdfReader(str(path)).pages)
                PDF_PAGE_COUNTS[str(path)] = n
                info.append(f"PDF with {n} pages")
                if n > LARGE_PDF_PAGES:
                    info.append(f"This is a LARGE PDF. Do NOT loop over its pages in "
                                f"run_python - that times out and returns nothing. Use "
                                f"find_pdf_pages to locate the table, then fetch_url for "
                                f"that narrow page range.")
            except Exception:
                pass
        info.append("This file stays on disk. In run_python, open it by PATH - "
                    "do NOT download it again.")
        return "\n".join(info)
    except Exception as e:
        return f"download failed: {e}"


def _limit_child_memory():
    """Run in the child before exec. A MemoryError in the child is recoverable;
    an OOM kill takes down the whole service and wipes every log with it."""
    try:
        import resource
        cap = CHILD_MEM_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except Exception:
        pass


def _child_env():
    """Environment for child processes.

    CRITICAL: we cap the child with RLIMIT_AS, which limits total VIRTUAL address
    space. OpenBLAS (pulled in by numpy, and therefore pandas) reserves a large
    slab of address space PER THREAD when it is imported, and blows straight
    through a 300MB cap - failing with 'OpenBLAS error: Memory allocation still
    failed after 10 retries' before any of our code runs. Pinning every math
    library to a single thread keeps that reservation small enough to import.
    Without this, pandas/openpyxl are unusable and no spreadsheet can be read."""
    env = dict(os.environ)
    env.update({
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    })
    return env


# numpy/pandas pull in OpenBLAS, which reserves a large chunk of VIRTUAL address
# space per worker thread on import. RLIMIT_AS caps virtual address space, so the
# two fight and pandas dies with "OpenBLAS error: Memory allocation still failed"
# before running a single line of the snippet. Pinning the thread pools to 1 keeps
# the reservation small enough to fit under the cap.
CHILD_ENV = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _run_child(code: str, timeout: int = 100) -> str:
    """Run `code` in its OWN process with a hard memory cap. This is what keeps
    PDF/Excel parsing from ever crashing the main bot process - a crash there
    would take down the Telegram poller and the log server with it, wiping
    every log in flight. Whatever the child print()s comes back to the agent."""
    timeout = min(timeout, _time_left(timeout))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout, cwd="/tmp",
            preexec_fn=_limit_child_memory, env=_child_env(),
        )
        out = (proc.stdout or "") + (("\nSTDERR:\n" + proc.stderr) if proc.stderr else "")
        if "MemoryError" in out:
            out += (f"\n[hit the {CHILD_MEM_MB}MB memory cap - try a narrower "
                    f"page range, or use find_pdf_pages to locate the right pages first]")
        return out.strip()[:15000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "timed out reading this file"
    except Exception as e:
        return f"execution failed: {e}"


def tool_fetch_url(url: str, max_chars: int = 20000, pages: str = "1-12") -> str:
    """Download a page or file and return readable text. `url` can be a web
    address OR a local path returned earlier by download_file - either works,
    and a local path is instant (no re-download).
    `pages` applies to PDFs only, e.g. "1-12" or "85-95". Government reports put
    their state-wise tables deep in the document - use find_pdf_pages first to
    locate the right page range instead of guessing, then come back for it.
    All PDF/Excel parsing runs in an isolated, memory-capped subprocess so a
    huge file can never crash the whole bot.
    """
    try:
        path, _ = _cached_download(url)
        is_pdf = path.suffix == ".pdf" or url.lower().endswith(".pdf")
        is_excel = path.suffix in (".xlsx", ".xls") or url.lower().endswith((".xlsx", ".xls"))

        if is_pdf:
            try:
                first, last = (int(x) for x in pages.split("-"))
            except ValueError:
                first, last = 1, 12
            last = min(last, first + 14)  # never more than 15 pages at once
            script = f"""
import pypdf
r = pypdf.PdfReader({str(path)!r})
total = len(r.pages)
first, last = {first}, min({last}, total)
print(f"[PDF has {{total}} pages; showing {{first}}-{{last}}]")
for i in range(first - 1, last):
    print(f"\\n--- page {{i + 1}} ---")
    print(r.pages[i].extract_text() or "")
"""
            return _run_child(script)[:max_chars]

        elif is_excel:
            script = f"""
import pandas as pd
sheets = pd.read_excel({str(path)!r}, sheet_name=None)
for n, d in sheets.items():
    print(f"### sheet: {{n}}")
    print(d.head(40).to_string())
"""
            return _run_child(script)[:max_chars]

        else:
            content = path.read_bytes()[:2_000_000]
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content.decode("utf-8", "replace"), "lxml")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            links = [f"{a.get_text(strip=True)} -> {a['href']}"
                    for a in soup.find_all("a", href=True)[:80]]
            text = soup.get_text("\n", strip=True)
            body = ((text + "\n\n--- LINKS ---\n" + "\n".join(links))
                    if len(text) > 200 else content.decode("utf-8", "replace"))
            return body[:max_chars]

    except MemoryError:
        return "ran out of memory reading this file. Try fewer PDF pages, or a CSV version."
    except Exception as e:
        return f"fetch failed: {e}"


# Text-extraction script for find_pdf_pages. Written as a plain template (not an
# f-string) so the embedded Python's braces don't need escaping. Placeholders are
# substituted with repr()'d values before it runs in a memory-capped subprocess.
#
# The important property: it works to a TIME BUDGET and always prints what it
# managed to find, then saves its progress. A 572-page government report cannot be
# text-extracted in one go on a free-tier box, so a scan that either finishes or
# returns nothing is useless. This one returns partial results and can be resumed.
_PDF_SCAN_SCRIPT = '''
import json, time
from pathlib import Path

pdf = __PDF_PATH__
kw = __KEYWORD__
budget = __BUDGET__

cache_path = Path(pdf + ".pages.json")
cache = {}
if cache_path.exists():
    try:
        cache = json.loads(cache_path.read_text())
    except Exception:
        cache = {}

engine = "pypdf"
try:
    import fitz  # PyMuPDF - roughly 10-30x faster than pypdf if installed
    doc = fitz.open(pdf)
    total = doc.page_count
    def get_text(i):
        return doc.load_page(i).get_text()
    engine = "pymupdf"
except Exception:
    import pypdf
    reader = pypdf.PdfReader(pdf)
    total = len(reader.pages)
    def get_text(i):
        return reader.pages[i].extract_text() or ""

start = time.time()
added = 0
MIN_PAGES = 25  # always make SOME progress, even if the budget is already blown -
                # otherwise repeated calls advance zero pages and loop forever
for i in range(total):
    key = str(i + 1)
    if key in cache:
        continue
    if added >= MIN_PAGES and time.time() - start > budget:
        break
    try:
        cache[key] = get_text(i)
    except Exception:
        cache[key] = ""
    added += 1

try:
    cache_path.write_text(json.dumps(cache))
except Exception as e:
    print("warning: could not save scan cache:", e)

hits = sorted(int(k) for k, v in cache.items() if kw in v.lower())
scanned = len(cache)

print("engine=" + engine + " scanned=" + str(scanned) + "/" + str(total)
      + " (this call added " + str(added) + ")")
if hits:
    print("pages containing keyword:", hits[:40])
    print("tip: now call fetch_url on this same file with a pages range around a hit, e.g. pages=330-344")
else:
    print("keyword not found in the " + str(scanned) + " pages scanned so far")

if scanned < total:
    print("NOT FINISHED - " + str(total - scanned) + " pages still unscanned. "
          "Call find_pdf_pages again with the SAME url to continue; pages already "
          "scanned are cached and cost nothing. Or try a shorter/different keyword.")
else:
    print("scan complete - the whole document has been checked.")
'''


def tool_find_pdf_pages(url: str, keyword: str) -> str:
    """Find which pages of a PDF mention a keyword. `url` can be a web address OR
    a local path from download_file.

    Built to survive huge reports: it extracts page text under a time budget,
    CACHES what it extracted to disk, and returns whatever it found so far. If it
    couldn't finish, call it again with the same url - it resumes from where it
    stopped and previously-scanned pages are instant. Never loop over all pages
    yourself in run_python; that just times out and returns nothing."""
    try:
        path, _ = _cached_download(url)
    except Exception as e:
        return f"scan failed: {e}"

    # Leave ~30s of the remaining time for the child to save its cache and for the
    # agent to still act on the result, rather than scanning right up to the wire.
    budget = max(10, min(PDF_SCAN_BUDGET, _time_left(PDF_SCAN_BUDGET) - 30))

    script = _PDF_SCAN_SCRIPT
    script = script.replace("__PDF_PATH__", repr(str(path)))
    script = script.replace("__KEYWORD__", repr(keyword.lower()))
    script = script.replace("__BUDGET__", str(budget))
    return _run_child(script, timeout=budget + 30)


# Matches a loop over EVERY page of a PDF: `for p in reader.pages`,
# `for i, p in enumerate(reader.pages)`, `range(len(reader.pages))`,
# `range(doc.page_count)`. A bounded slice like `reader.pages[60:75]` won't match,
# so reading a known range is still allowed.
_FULL_PAGE_LOOP = re.compile(
    r"(?:for\s+[\w,\s]+\s+in\s+(?:enumerate\s*\()?\s*[\w\.]*\.pages\b(?!\s*\[)"
    r"|range\s*\(\s*len\s*\(\s*[\w\.]*\.pages\b(?!\s*\[)"
    r"|range\s*\(\s*[\w\.]*\.page_count\b)")


def _blocked_full_page_scan(code: str):
    """If this snippet would iterate every page of a PDF we know to be large,
    return an explanation. This is enforced in CODE because the system prompt
    already forbids it and the model does it anyway - and it costs 120s of a
    240s budget to learn that lesson each time."""
    if not _FULL_PAGE_LOOP.search(code):
        return None
    for quoted in re.findall(r"['\"]([^'\"]+\.pdf)['\"]", code):
        pages = PDF_PAGE_COUNTS.get(quoted)
        if pages and pages > LARGE_PDF_PAGES:
            return (f"Refused before running: this loops over all {pages} pages of "
                    f"{quoted}. On a document this size that hits the timeout and "
                    f"returns you NOTHING - it has already cost this run dearly.\n"
                    f"Do this instead:\n"
                    f"  1. find_pdf_pages(url=<the same url or path>, keyword=...) "
                    f"to get the page numbers - it is cached and takes seconds.\n"
                    f"  2. fetch_url(url=..., pages=\"N-M\") for just those pages.\n"
                    f"If you already know the page numbers, read them directly with a "
                    f"bounded slice, e.g. reader.pages[200:209].")
    return None


def tool_run_python(code: str) -> str:
    """Run Python in a subprocess. Whatever you print() comes back to you."""
    blocked = _blocked_full_page_scan(code)
    if blocked:
        return blocked
    budget = min(120, _time_left(120))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=budget, cwd="/tmp",
            preexec_fn=_limit_child_memory, env=_child_env(),
        )
        out = (proc.stdout or "") + (("\nSTDERR:\n" + proc.stderr) if proc.stderr else "")
        if proc.returncode != 0 and ("NameError" in out or "ModuleNotFoundError" in out):
            out += ("\n[each run_python call is a FRESH process - re-import every module "
                    "and re-create every variable inside this snippet. Downloaded files "
                    "in /tmp do persist.]")
        if proc.returncode != 0 and "MemoryError" in out:
            out += (f"\n[hit the {CHILD_MEM_MB}MB memory cap - read the file in chunks "
                    f"with pandas' chunksize, or select fewer columns]")
        return out.strip()[:15000] or "(no output - remember to print())"
    except subprocess.TimeoutExpired:
        return (f"code timed out after {budget}s. If you were scanning a large PDF, "
                f"use find_pdf_pages instead - it caches progress and returns partial results.")
    except Exception as e:
        return f"execution failed: {e}"


TOOL_SCHEMA = [
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the web for datasets, pages or facts.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "reason": {"type": "string", "description": "one line: why you are doing this"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "find_pdf_pages",
        "description": ("Scan a PDF for a keyword and return which page numbers mention it. "
                        "Use this on any large report BEFORE fetch_url, instead of guessing "
                        "a page range."),
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"},
            "keyword": {"type": "string"}},
            "required": ["url", "keyword"]}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": ("Download a URL (HTML, CSV, JSON, PDF or Excel) and read it as text. "
                        "For PDFs use `pages` to pick a range, e.g. \"85-95\" - use "
                        "find_pdf_pages first to find where the table actually is."),
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"},
            "pages": {"type": "string", "description": "PDF page range, default \"1-12\", max 15 pages"},
            "reason": {"type": "string", "description": "one line: why you are doing this"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "download_file",
        "description": ("Download a big file ONCE and keep it on disk. Returns the local path "
                        "and, for PDFs, the page count. Use this before run_python for any "
                        "large PDF/Excel/CSV, then open it by path."),
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"},
            "reason": {"type": "string", "description": "one line: why you are doing this"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": ("Run Python code and get whatever it prints. "
                        "pandas, numpy, requests, bs4, openpyxl, pypdf are installed. "
                        "Use this for all real computation."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"},
            "reason": {"type": "string", "description": "one line: why you are doing this"}},
            "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Submit the final answer. Call this exactly once, at the end.",
        "parameters": {"type": "object", "properties": {
            "answer_json": {
                "type": "string",
                "description": ("The value for the \"answer\" key, as a JSON string. "
                                "Must match the shape the question asked for. "
                                "Example: {\"state\": \"Assam\"}")}},
            "required": ["answer_json"]}}},
]

TOOL_FUNCS = {
    "search_web": tool_search_web,
    "download_file": tool_download_file,
    "fetch_url": tool_fetch_url,
    "find_pdf_pages": tool_find_pdf_pages,
    "run_python": tool_run_python,
}

# Models improvise on the schema. Map the common inventions onto real parameters.
ALIASES = {
    "search_web": {"queries": "query", "q": "query", "search_query": "query",
                   "text": "query", "keywords": "query"},
    "download_file": {"urls": "url", "link": "url", "file_url": "url"},
    "fetch_url": {"urls": "url", "link": "url", "page": "pages",
                  "page_range": "pages", "page_numbers": "pages"},
    "find_pdf_pages": {"urls": "url", "link": "url", "word": "keyword", "term": "keyword"},
    "run_python": {"python": "code", "script": "code", "source": "code",
                   "python_code": "code"},
}


def call_tool(name, args, log, step):
    """Never let a malformed tool call end the run. Hand the model the error
    instead - it can usually correct itself on the next step."""
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return f"unknown tool '{name}'. Available: {', '.join(TOOL_FUNCS)}"
    args = dict(args or {})
    reason = args.pop("reason", None)
    if reason:
        log.write("tool_reason", step=step, tool=name, reason=str(reason)[:500])
    for wrong, right in ALIASES.get(name, {}).items():
        if wrong in args and right not in args:
            args[right] = args.pop(wrong)
    accepted = set(inspect.signature(fn).parameters)
    unexpected = [k for k in args if k not in accepted]
    if unexpected:
        log.write("tool_args_dropped", step=step, tool=name, dropped=unexpected)
        args = {k: v for k, v in args.items() if k in accepted}
    try:
        return fn(**args)
    except TypeError as e:
        return (f"wrong arguments for {name}: {e}. "
                f"This tool accepts: {sorted(accepted)}. Try again.")
    except Exception as e:
        return f"{name} failed: {type(e).__name__}: {e}"


SYSTEM_PROMPT = """You are a careful data analyst agent.

You are given a data-analysis question. Work out the real answer using your tools.

Rules:
- Never guess a number from memory. Find the source, download it, compute with run_python.
- Search results and news articles are NOT the source of truth - they often disagree with
  each other and with the real report. For any question about a specific report or dataset
  (PLFS, MOSPI, RBI, Census, etc.), you must open the primary document itself with
  find_pdf_pages / fetch_url / download_file before you can submit an answer. Search is only
  for locating the document's URL, never for reading off the final number.
- run_python is STATELESS. Each call is a brand new process: imports, variables and
  downloads from a previous call are GONE. Every snippet must import what it needs.
- NEVER download the same file twice. Use download_file once - it saves to /tmp and
  files there DO persist between run_python calls - then open it by path.
- NEVER loop over every page of a large PDF inside run_python. On a 500+ page
  report that hits the 120s timeout and returns you NOTHING - pure wasted time.
  Use find_pdf_pages instead: it works to a time budget, caches its progress, and
  always returns partial results. If it says NOT FINISHED, just call it again with
  the same url to continue where it stopped.
- For a big PDF: download_file it ONCE, then reuse it by passing the SAME url or the
  local path it gives you back to find_pdf_pages / fetch_url - never re-download.
  Use find_pdf_pages to find which pages mention the state/table/keyword you need,
  then fetch_url just that narrow range. Never guess a page range in a long report.
- Prefer a CSV or Excel version of a dataset over a large PDF whenever one exists.
  data.gov.in often has the same table as a clean CSV. MoSPI in particular publishes
  each appendix table of a report as its OWN small spreadsheet under
  api.mospi.gov.in/api/esankhyiki/file/download/datacatalogue/<PRODUCT>/<YEAR>/Table_<N>.xlsx
  - if you know the table number you need (the report's Appendix A contents page
  lists them), fetch that one small file instead of the whole report. Reading a
  300KB spreadsheet beats hunting through a 500-page PDF every time.
- A full "Annual Report" is often 300-600 pages and slow to search. Before opening
  one, check search results for a SMALLER, more targeted official document that
  already contains just the table you need - e.g. a PIB press release, a
  parliament (Lok Sabha/Rajya Sabha, sansad.in) reply, or a ministry press note.
  These are usually a few pages and often already have the exact state-wise
  breakdown you're looking for. Only fall back to the full annual report if
  nothing smaller has the answer.
- Search is for LOCATING a document, not for reading its content. As soon as a
  search result gives you a plausible official PDF/document link, stop searching
  and open it - do not run several more searches "to confirm" a link you already have.
  You have a limited number of searches per question; spend them on finding new
  leads, not rephrasing the same query.
- Do not rely on "site:domain.com" search filters - this search tool does not
  reliably support them and can return unrelated results. Use plain keyword
  searches instead, and look for official domains in the results yourself.
- Government PDFs are sometimes bilingual (Hindi followed by English, or vice
  versa). If your keyword search matches too many pages or none, try the
  English name/spelling and check whether the document has a separate English
  section rather than scanning the whole thing indiscriminately.
- Prefer official sources (mospi.gov.in, pib.gov.in, sansad.in, data.gov.in, RBI,
  Census, World Bank, etc.).
- If data is embedded in the question itself, just compute on it directly - no need to search
  or open any document.
- Read the question's JSON template carefully. Your answer must match that shape EXACTLY:
  same keys, same spelling, same data types. If it asks for a number, give a number, not a string.
- Return EXACTLY the keys asked for and NO others. If the question asks only for "sum",
  do not also include mean, median, min or max. Extra keys are marked wrong.
- submit_answer takes ONLY the inner value. If the template is
  {"answer": {"state": "..."}, "log_url": "..."}, you submit {"state": "..."} - not the
  whole envelope, and never a "log_url" key.
- Earlier messages are context only. If a question refers to data given "at the start",
  it means the start of THIS exchange - never data from an older, unrelated question.
- Work efficiently. Every tool call costs a round trip, so do not use run_python for
  trivial arithmetic you can do reliably in your head. Use it for real data work:
  parsing files, aggregating rows, sorting, statistics.
- Batch your work: if you need three numbers from one file, get them in one run_python
  call, not three.
- Do not re-verify an answer you are already confident in.
- When you are confident, call submit_answer once. That ends the run.
"""

# --------------------------------------------------------------------------
# The agent loop
# --------------------------------------------------------------------------
def run_agent(question: str, log: RunLog):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    log.write("question", text=question)
    started = time.time()
    deadline = started + TIME_BUDGET
    _ctx.deadline = deadline  # tools clamp their own timeouts against this

    # Code-enforced guardrail: don't let the agent answer a question it went
    # searching for using ONLY search snippets. Search results routinely
    # disagree with each other and with the real report (this is exactly what
    # made the bot answer "Nagaland" once and "Goa" another time for the same
    # question). This is enforced here, in code, because the prompt text
    # above is not something the grader lets us guarantee will be followed.
    used_search = False
    used_real_source = False
    submit_blocks = 0
    MAX_SUBMIT_BLOCKS = 2  # don't loop forever if the model can't find a working source

    search_calls = 0
    SEARCH_CALL_LIMIT = int(os.environ.get("SEARCH_CALL_LIMIT", 6))
    candidate_urls: list[str] = []

    empty_turns = 0
    MAX_EMPTY_TURNS = 8  # observed 7 empty turns in a single run - null usage
                         # fields, no tool call, no text. Charging those against
                         # the step budget cost 29% of it.
    warned_time_up = False
    step = -1
    while step + 1 < MAX_STEPS:
        step += 1
        out_of_time = time.time() > deadline
        if out_of_time and not warned_time_up:
            warned_time_up = True  # say it once; repeating it every turn just burns context
            messages.append({"role": "user",
                             "content": "Time is up. Call submit_answer NOW with your best "
                             "answer from what you already know. No more tools."})
        t0 = time.time()
        resp = llm(messages, log, step, deadline)
        msg = resp.choices[0].message

        # The model's actual thinking trace, if the provider returned one. This is
        # NOT the same thing as msg.content (which is just the text it wrote next to
        # its tool call, and is usually empty). Kept separate so the log tells you
        # honestly whether reasoning happened.
        thinking = getattr(msg, "reasoning", None)
        if thinking is None and getattr(msg, "model_extra", None):
            thinking = msg.model_extra.get("reasoning")

        dumped = msg.model_dump(exclude_none=True)
        # Don't echo provider-specific reasoning fields back in the next request -
        # some providers reject unknown fields on input.
        for k in ("reasoning", "reasoning_details"):
            dumped.pop(k, None)
        messages.append(dumped)

        usage = getattr(resp, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None)

        log.write("model_turn", step=step,
                  seconds=round(time.time() - t0, 2),
                  prompt_tokens=getattr(usage, "prompt_tokens", None),
                  completion_tokens=getattr(usage, "completion_tokens", None),
                  reasoning_tokens=reasoning_tokens,
                  thinking=(str(thinking)[:2000] if thinking else None),
                  content=(msg.content or "")[:2000],
                  tool_calls=[tc.function.name for tc in (msg.tool_calls or [])])

        if not msg.tool_calls:
            # A completely empty turn (no tool call, no text - it happens, and the
            # usage fields come back null) is an API hiccup, not a decision. Don't
            # charge it against the step budget; just nudge and retry.
            if not (msg.content or "").strip() and empty_turns < MAX_EMPTY_TURNS:
                empty_turns += 1
                step -= 1
                log.write("empty_turn", step=step, count=empty_turns)
            messages.append({"role": "user",
                             "content": "Use a tool, or call submit_answer if you are done."})
            continue

        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "submit_answer":
                block = (used_search and not used_real_source
                         and submit_blocks < MAX_SUBMIT_BLOCKS and not out_of_time)
                if block:
                    submit_blocks += 1
                    log.write("submit_blocked", step=step,
                              reason="no primary source opened yet", attempt=submit_blocks)
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": "Not accepted: you've only used search_web so "
                                     "far. Search snippets often disagree with each other and "
                                     "with the real report. Use find_pdf_pages, fetch_url, or "
                                     "download_file to actually open the primary source and "
                                     "find the real answer, then call submit_answer again."})
                    continue
                raw = args.get("answer_json", "")
                answer = unwrap(coerce_json(raw))
                log.write("submit_answer", raw=raw, parsed=answer)
                return answer

            log.write("tool_call", step=step, tool=name, args=args)
            if out_of_time:
                log.write("tool_refused", step=step, tool=name, reason="past deadline")
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": "DEADLINE PASSED. No more tools. "
                                 "Call submit_answer immediately."})
                continue

            if name == "search_web":
                if search_calls >= SEARCH_CALL_LIMIT and candidate_urls:
                    log.write("search_blocked", step=step, calls_used=search_calls,
                              candidates=candidate_urls[:5])
                    hint = "\n".join(f"- {u}" for u in candidate_urls[:5])
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": f"Not run: you've already used {search_calls} "
                                     f"searches and have promising official links from earlier "
                                     f"results. Stop searching and open one of these directly "
                                     f"with download_file or fetch_url instead:\n{hint}"})
                    continue
                search_calls += 1
                used_search = True

            t1 = time.time()
            result = call_tool(name, args, log, step)

            if name == "search_web":
                for u in _extract_candidate_urls(str(result)):
                    if u not in candidate_urls:
                        candidate_urls.append(u)
                if str(result).startswith(OFF_TOPIC_MARKER):
                    # A backend fault is not the agent's fault. Don't charge it
                    # against SEARCH_CALL_LIMIT - in one observed run 3 of 6
                    # searches were junk, and the agent was then blocked from
                    # searching with almost nothing usable to show for it.
                    search_calls = max(0, search_calls - 1)
                    log.write("search_refunded", step=step, calls_used=search_calls)

            if name in ("fetch_url", "download_file", "find_pdf_pages"):
                failed = str(result).lower().startswith(
                    ("fetch failed", "download failed", "scan failed"))
                if not failed:
                    used_real_source = True

            log.write("tool_result", step=step, tool=name,
                      seconds=round(time.time() - t1, 2),
                      chars=len(str(result)),
                      result=str(result)[:4000])
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})

    if used_real_source:
        log.write("give_up_retry", reason="step limit reached but a real source was opened")
        messages.append({"role": "user",
                         "content": "You are out of steps. Call submit_answer NOW with your "
                         "best answer based on whatever you have already read. No more tools."})
        try:
            resp = llm(messages, log, MAX_STEPS, deadline)
            msg = resp.choices[0].message
            if msg.tool_calls:
                for call in msg.tool_calls:
                    if call.function.name == "submit_answer":
                        args = json.loads(call.function.arguments or "{}")
                        raw = args.get("answer_json", "")
                        answer = unwrap(coerce_json(raw))
                        log.write("submit_answer", raw=raw, parsed=answer, forced=True)
                        return answer
        except Exception as e:
            log.write("give_up_retry_failed", error=str(e))

    log.write("give_up", reason="step limit reached")
    return {"error": "could not determine answer"}


def unwrap(value):
    """The model sometimes submits the whole envelope by mistake, e.g.
    {"answer": {"region": "East"}} when it should have submitted just
    {"region": "East"}. Peel that off. Also drops a stray log_url."""
    for _ in range(3):
        if not isinstance(value, dict):
            break
        if set(value) <= {"answer", "log_url"} and isinstance(value.get("answer"), (dict, list)):
            value = value["answer"]
            continue
        if "log_url" in value and len(value) > 1:
            value = {k: v for k, v in value.items() if k != "log_url"}
        break
    return value


def coerce_json(raw: str):
    """The model sometimes wraps JSON in code fences or prose. Dig it out."""
    if isinstance(raw, (dict, list, int, float)):
        return raw
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return text


# --------------------------------------------------------------------------
# Telegram plumbing
# --------------------------------------------------------------------------
def tg_send(chat_id: int, text: str):
    try:
        requests.post(f"{TG_API}/sendMessage",
                      json={"chat_id": chat_id, "text": text}, timeout=30)
    except Exception:
        traceback.print_exc()


def handle_message(chat_id: int, text: str, sent_at: float | None = None):
    log = RunLog()
    picked_up = time.time()
    if sent_at:
        log.write("delivery", telegram_sent_at=sent_at,
                  seconds_before_bot_saw_it=round(picked_up - sent_at, 2))
    try:
        past = history.get(chat_id, [])[-5:-1]
        question = text
        if past:
            context = "\n".join(f"- {m}" for m in past)
            question = (f"Earlier messages in this conversation (context only):\n{context}\n\n"
                       f"ANSWER THIS MESSAGE:\n{text}")
        answer = run_agent(question, log)
    except Exception as e:
        traceback.print_exc()
        log.write("error", error=str(e), trace=traceback.format_exc())
        answer = {"error": str(e)}

    agent_done = time.time()
    log.write("agent_done", agent_seconds=round(agent_done - picked_up, 2))
    reply = json.dumps({"answer": answer, "log_url": log.public_url}, ensure_ascii=False)
    log.write("reply", text=reply)
    log.publish()  # upload the COMPLETE log, reply included
    tg_send(chat_id, reply)
    if "log_url" in text:
        history[chat_id] = []  # that message ended the exchange
    log.write("sent", send_seconds=round(time.time() - agent_done, 2),
              total_seconds=round(time.time() - (sent_at or picked_up), 2))


def chat_worker(chat_id: int, q):
    """One worker per chat. Messages are answered strictly in the order they
    arrived, so a fast question can never overtake a slow one."""
    while True:
        text, sent_at = q.get()
        try:
            age = time.time() - sent_at if sent_at else 0
            if age > STALE_AFTER:
                # The grader has already given up on this one. Replying now would
                # land our answer inside the NEXT question's conversation.
                print(f"[{chat_id}] dropping stale message ({age:.0f}s old)", flush=True)
                continue
            now = time.time()
            if now - last_seen.get(chat_id, 0) > CONV_GAP:
                history[chat_id] = []  # long gap = new question
            last_seen[chat_id] = now
            history.setdefault(chat_id, []).append(text)
            handle_message(chat_id, text, sent_at)
        except Exception:
            traceback.print_exc()
        finally:
            q.task_done()


def poll_telegram():
    print("Telegram polling started", flush=True)
    queues: dict[int, "queue.Queue"] = {}
    offset = None
    while True:
        try:
            r = requests.get(f"{TG_API}/getUpdates",
                             params={"timeout": 50, "offset": offset}, timeout=70)
            for update in r.json().get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message") or update.get("channel_post") or {}
                text = msg.get("text")
                chat_id = msg.get("chat", {}).get("id")
                if not text or not chat_id:
                    continue
                print(f"[{chat_id}] {text[:120]}", flush=True)
                if chat_id not in queues:
                    queues[chat_id] = queue.Queue()
                    threading.Thread(target=chat_worker, args=(chat_id, queues[chat_id]),
                                    daemon=True).start()
                queues[chat_id].put((text, msg.get("date")))
        except Exception:
            traceback.print_exc()
        time.sleep(3)


# --------------------------------------------------------------------------
# Web server - keeps the host awake and serves the logs
# --------------------------------------------------------------------------
app = Flask(__name__)


@app.get("/")
def home():
    return jsonify(status="ok", logs=len(list(LOG_DIR.glob("*.jsonl"))))


@app.get("/logs/<path:filename>")
def serve_log(filename):
    return send_from_directory(LOG_DIR, filename, mimetype="application/x-ndjson")


if __name__ == "__main__":
    threading.Thread(target=poll_telegram, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)
