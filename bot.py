"""
Data Analyst Telegram Bot
-------------------------
One process that does three things:
  1. Serves a tiny website (so your run logs have a public URL)
  2. Polls Telegram for new messages
  3. Runs an LLM agent that can search the web, download files and run Python

Run locally:  python bot.py
"""

import os
import re
import io
import json
import sys
import time
import uuid
import threading
import subprocess
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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

TG_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

MAX_STEPS = 14           # how many tool calls the agent may make
TIME_BUDGET = 210        # seconds before we force an answer

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Remembers the last few messages per chat, for multi-turn questions
history: dict[int, list[str]] = {}
last_seen: dict[int, float] = {}
CONV_GAP = 240          # a gap this long means a NEW conversation started


def llm(messages, log, step):
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


# --------------------------------------------------------------------------
# TOOLS - the actions the agent is allowed to take
# --------------------------------------------------------------------------
def tool_search_web(query: str) -> str:
    """Search the web and return titles, links and snippets."""
    try:
        from ddgs import DDGS
        with DDGS() as ddg:
            hits = list(ddg.text(query, max_results=8))
        if not hits:
            return "No results."
        return "\n".join(
            f"- {h.get('title')}\n  {h.get('href')}\n  {h.get('body','')[:300]}"
            for h in hits
        )
    except Exception as e:
        return f"search failed: {e}"


def tool_fetch_url(url: str, max_chars: int = 20000) -> str:
    """Download a page or file and return readable text."""
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        ctype = r.headers.get("content-type", "").lower()

        if "pdf" in ctype or url.lower().endswith(".pdf"):
            import pdfplumber
            text = []
            with pdfplumber.open(io.BytesIO(r.content)) as pdf:
                for page in pdf.pages[:25]:
                    text.append(page.extract_text() or "")
            body = "\n".join(text)

        elif any(x in ctype for x in ("excel", "spreadsheet")) or url.lower().endswith((".xlsx", ".xls")):
            import pandas as pd
            sheets = pd.read_excel(io.BytesIO(r.content), sheet_name=None)
            body = "\n\n".join(f"### sheet: {n}\n{d.head(40).to_string()}" for n, d in sheets.items())

        elif "html" in ctype:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            links = [f"{a.get_text(strip=True)} -> {a['href']}"
                     for a in soup.find_all("a", href=True)[:80]]
            body = soup.get_text("\n", strip=True) + "\n\n--- LINKS ---\n" + "\n".join(links)
        else:
            body = r.text

        return body[:max_chars]
    except Exception as e:
        return f"fetch failed: {e}"


def tool_run_python(code: str) -> str:
    """Run Python in a subprocess. Whatever you print() comes back to you."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120, cwd="/tmp",
        )
        out = (proc.stdout or "") + (("\nSTDERR:\n" + proc.stderr) if proc.stderr else "")
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
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Download a URL (HTML, CSV, JSON, PDF or Excel) and read it as text.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": ("Run Python code and get whatever it prints. "
                        "pandas, numpy, requests, bs4, openpyxl, pdfplumber are installed. "
                        "Use this for all real computation."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}}, "required": ["code"]}}},
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
    "fetch_url": tool_fetch_url,
    "run_python": tool_run_python,
}

SYSTEM_PROMPT = """You are a careful data analyst agent.

You are given a data-analysis question. Work out the real answer using your tools.

Rules:
- Never guess a number from memory. Find the source, download it, compute with run_python.
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

    for step in range(MAX_STEPS):
        if time.time() - started > TIME_BUDGET:
            messages.append({"role": "user",
                             "content": "Time is up. Call submit_answer now with your best answer."})

        t0 = time.time()
        resp = llm(messages, log, step)
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
            t1 = time.time()
            result = TOOL_FUNCS.get(name, lambda **k: "unknown tool")(**args)
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
    reply = json.dumps({"answer": answer, "log_url": log.url}, ensure_ascii=False)
    log.write("reply", text=reply, agent_seconds=round(agent_done - picked_up, 2))
    tg_send(chat_id, reply)
    if "log_url" in text:
        history[chat_id] = []       # that message ended the exchange
    log.write("sent", send_seconds=round(time.time() - agent_done, 2),
              total_seconds=round(time.time() - (sent_at or picked_up), 2))


def poll_telegram():
    print("Telegram polling started", flush=True)
    pool = ThreadPoolExecutor(max_workers=4)
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
                now = time.time()
                if now - last_seen.get(chat_id, 0) > CONV_GAP:
                    history[chat_id] = []          # long gap = new question
                last_seen[chat_id] = now
                history.setdefault(chat_id, []).append(text)
                pool.submit(handle_message, chat_id, text, msg.get("date"))
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
