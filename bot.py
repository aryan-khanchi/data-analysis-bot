"""
Data Analyst Telegram Bot
-------------------------
One process that does three things:
  1. Serves a tiny website (so your run logs have a public URL)
  2. Polls Telegram for new messages
  3. Runs an LLM agent that can search the web, download files and run Python

Run locally:  python bot.py
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

try:                                  # loads your .env file when running locally
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
GITHUB_REPO = os.environ.get("GITHUB_REPO")        # e.g. "avi/tds-telegram-bot"
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

# Memory guards - the free tier gives us 512MB for everything
MAX_DOWNLOAD = int(os.environ.get("MAX_DOWNLOAD_MB", 8)) * 1_000_000
CHILD_MEM_MB = int(os.environ.get("CHILD_MEM_MB", 300))

TG_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

MAX_STEPS = int(os.environ.get("MAX_STEPS", 16))
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", 240))   # grader allows 300s

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Remembers the last few messages per chat, for multi-turn questions
history: dict[int, list[str]] = {}
last_seen: dict[int, float] = {}
CONV_GAP = 240          # a gap this long means a NEW conversation started
STALE_AFTER = 300       # older than the grader's timeout - answering it can only cause harm


def llm(messages, log, step, deadline=None):
    """Call the model, retrying politely when we hit the rate limit."""
    last_error = None
    for attempt in range(6):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOL_SCHEMA, temperature=0,
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
def tool_search_web(query) -> str:
    """Search the web. Accepts one query or a list of them."""
    queries = query if isinstance(query, list) else [query]
    out = []
    try:
        from ddgs import DDGS
        with DDGS() as ddg:
            for q in queries[:3]:
                hits = list(ddg.text(str(q), max_results=8))
                out.append(f"### results for: {q}")
                out.extend(f"- {h.get('title')}\n  {h.get('href')}\n  {h.get('body','')[:300]}"
                           for h in hits) if hits else out.append("(no results)")
        return "\n".join(out)
    except Exception as e:
        return f"search failed: {e}"


DOWNLOAD_DIR = Path("/tmp/downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _cached_download(url: str):
    """Fetch a URL to disk once. Later calls for the same URL are free.
    Returns (path, from_cache) or raises."""
    ext = ""
    for e in (".pdf", ".xlsx", ".xls", ".csv", ".json", ".zip"):
        if url.lower().split("?")[0].endswith(e):
            ext = e
            break
    path = DOWNLOAD_DIR / (hashlib.sha1(url.encode()).hexdigest()[:16] + ext)
    if path.exists() and path.stat().st_size > 0:
        return path, True

    r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
    r.raise_for_status()
    size = 0
    with path.open("wb") as f:                 # straight to disk, never all in RAM
        for chunk in r.iter_content(65536):
            size += len(chunk)
            if size > MAX_DOWNLOAD:
                f.close()
                path.unlink(missing_ok=True)
                raise ValueError(f"file exceeds {MAX_DOWNLOAD // 1_000_000}MB")
            f.write(chunk)
    return path, False


def tool_download_file(url: str) -> str:
    """Download once, keep it on disk, and report what's inside."""
    try:
        path, cached = _cached_download(url)
        size_mb = path.stat().st_size / 1e6
        note = "already downloaded earlier" if cached else "downloaded"
        info = [f"{note}: {path}  ({size_mb:.1f} MB)"]

        if path.suffix == ".pdf":
            try:
                from pypdf import PdfReader
                info.append(f"PDF with {len(PdfReader(str(path)).pages)} pages")
            except Exception:
                pass
        info.append("This file stays on disk. In run_python, open it by PATH - "
                    "do NOT download it again.")
        return "\n".join(info)
    except Exception as e:
        return f"download failed: {e}"


def tool_fetch_url(url: str, max_chars: int = 20000, pages: str = "1-12") -> str:
    """Download a page or file and return readable text.

    `pages` applies to PDFs only, e.g. "1-12" or "85-95". Government reports put
    their state-wise tables deep in the document, so read the contents page
    first, then come back for the range you actually need. Keep ranges small -
    this runs on a 512MB box.
    """
    try:
        path, _ = _cached_download(url)
        ctype = ("pdf" if path.suffix == ".pdf" else
                 "excel" if path.suffix in (".xlsx", ".xls") else "")

        if ctype == "pdf" or url.lower().endswith(".pdf"):
            try:
                first, last = (int(x) for x in pages.split("-"))
            except ValueError:
                first, last = 1, 12
            last = min(last, first + 14)          # never more than 15 pages at once
            import pdfplumber
            out = []
            with pdfplumber.open(str(path)) as pdf:
                total = len(pdf.pages)
                out.append(f"[PDF has {total} pages; showing {first}-{min(last, total)}]")
                for i in range(first - 1, min(last, total)):
                    page = pdf.pages[i]
                    out.append(f"\n--- page {i + 1} ---\n" + (page.extract_text() or ""))
                    flush = getattr(page, "flush_cache", None)
                    if flush:
                        flush()               # release the page before loading the next
            body = "\n".join(out)

        elif any(x in ctype for x in ("excel", "spreadsheet")) or url.lower().endswith((".xlsx", ".xls")):
            import pandas as pd
            sheets = pd.read_excel(str(path), sheet_name=None)
            body = "\n\n".join(f"### sheet: {n}\n{d.head(40).to_string()}" for n, d in sheets.items())

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


def _limit_child_memory():
    """Run in the child before exec. A MemoryError in the child is recoverable;
    an OOM kill takes down the whole service and wipes every log with it."""
    try:
        import resource
        cap = CHILD_MEM_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except Exception:
        pass


def tool_run_python(code: str) -> str:
    """Run Python in a subprocess. Whatever you print() comes back to you."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120, cwd="/tmp",
            preexec_fn=_limit_child_memory,
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
        return "code timed out after 120s"
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
        "name": "fetch_url",
        "description": ("Download a URL (HTML, CSV, JSON, PDF or Excel) and read it as text. "
                        "For PDFs use `pages` to pick a range, e.g. \"85-95\" - read the "
                        "contents page first to find where the table is."),
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
                        "pandas, numpy, requests, bs4, openpyxl, pdfplumber are installed. "
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
    "run_python": tool_run_python,
}

# Models improvise on the schema. Map the common inventions onto real parameters.
ALIASES = {
    "search_web": {"queries": "query", "q": "query", "search_query": "query",
                   "text": "query", "keywords": "query"},
    "download_file": {"urls": "url", "link": "url", "file_url": "url"},
    "fetch_url": {"urls": "url", "link": "url", "page": "pages",
                  "page_range": "pages", "page_numbers": "pages"},
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
- run_python is STATELESS. Each call is a brand new process: imports, variables and
  downloads from a previous call are GONE. Every snippet must import what it needs.
- NEVER download the same file twice. Use download_file once - it saves to /tmp and
  files there DO persist between run_python calls - then open it by path.
- For a big PDF: download_file first (it tells you the page count), then read a NARROW
  page range. Use pypdf for plain text (light on memory); use pdfplumber only for
  specific pages you need tables from. Never loop pdfplumber over hundreds of pages -
  it will run out of memory.
- Prefer a CSV or Excel version of a dataset over a large PDF whenever one exists.
  data.gov.in often has the same table as a clean CSV.
- Prefer official sources (mospi.gov.in, data.gov.in, RBI, Census, World Bank, etc.).
- If data is embedded in the question itself, just compute on it directly - no need to search.
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

    for step in range(MAX_STEPS):
        out_of_time = time.time() > deadline
        if out_of_time:
            messages.append({"role": "user",
                             "content": "Time is up. Call submit_answer NOW with your best "
                                        "answer from what you already know. No more tools."})

        t0 = time.time()
        resp = llm(messages, log, step, deadline)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        usage = getattr(resp, "usage", None)
        log.write("model_turn", step=step,
                  seconds=round(time.time() - t0, 2),
                  prompt_tokens=getattr(usage, "prompt_tokens", None),
                  completion_tokens=getattr(usage, "completion_tokens", None),
                  reasoning=(msg.content or "")[:2000],
                  tool_calls=[tc.function.name for tc in (msg.tool_calls or [])])

        if not msg.tool_calls:
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
            t1 = time.time()
            result = call_tool(name, args, log, step)
            log.write("tool_result", step=step, tool=name,
                      seconds=round(time.time() - t1, 2),
                      chars=len(str(result)),
                      result=str(result)[:4000])
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})

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
    log.publish()          # upload the COMPLETE log, reply included
    tg_send(chat_id, reply)
    if "log_url" in text:
        history[chat_id] = []       # that message ended the exchange
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
                history[chat_id] = []          # long gap = new question
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
