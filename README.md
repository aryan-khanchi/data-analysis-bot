# Data Analyst Telegram Bot

An LLM agent on Telegram. Send it a data-analysis question, it researches the answer
and replies with a single JSON object:

```json
{"answer": {"state": "Assam"}, "log_url": "https://your-app.onrender.com/logs/ab12cd34.jsonl"}
```

## How it works

One Python process does three jobs:

1. **Flask web server** - serves the run logs at a public URL, and gives the host
   something to health-check.
2. **Telegram poller** - long-polls for new messages, one worker thread per message.
3. **Agent loop** - an LLM with four tools: `search_web`, `fetch_url`, `run_python`,
   and `submit_answer`. It keeps calling tools until it has a real answer.

Every step is appended to `logs/<run_id>.jsonl`, which is what `log_url` points to.

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in your keys
python bot.py
```

## Deploy (Render)

Push to GitHub, create a Web Service from the repo, add the environment variables
from `.env.example`, and set `PUBLIC_BASE_URL` to the URL Render gives you.

## Files

| File | Purpose |
|---|---|
| `bot.py` | Everything: server, poller, agent, tools |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for your secrets |
| `render.yaml` | Deployment config |
