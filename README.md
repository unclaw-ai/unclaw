<p align="center">
  <img src="./unclaw-animated.svg" alt="Unclaw animated logo" />
</p>

<h1 align="center">Unclaw 🦐</h1>

<p align="center"><strong>Local-first AI agent/runtime for real local models.</strong></p>

<p align="center">No cloud. No lock-in. No unnecessary framework bloat.</p>

Unclaw is a lightweight local-first runtime for AI models served through Ollama.

Today it ships as a shared local runtime with terminal and Telegram channels, safe local tools, grounded web search, session persistence, onboarding, and local logs/traces.

It also ships a bounded agent loop. The default `main` profile stays conservative, while the shipped `deep` profile enables native tool calling.

---

## Why Unclaw

Most “AI agent” projects assume cloud APIs, huge frameworks, and expensive hosted models.

Unclaw takes the opposite path:

- **local-first**
- **privacy-first**
- **secure-by-default**
- **lightweight**
- **open source**
- **built for real local models and real computers**

---

## Project status

### Implemented now

- Local Ollama runtime with `fast`, `main`, `deep`, and `codex` profiles
- Shipped `deep` profile configured for native tool calling; other shipped profiles remain conservative `json_plan` profiles
- Interactive terminal via `unclaw start`
- Telegram polling bot with local allowlist management
- Session persistence plus deterministic session summaries
- Slash commands for `/read`, `/ls`, `/fetch`, and grounded `/search`
- Model-assisted routing between normal chat and web-backed search
- Grounded search replies with compact sources and grounded follow-up turns
- `search_web` integrated into the shared runtime path: non-native profiles use a runtime pre-executed search step, while native profiles can call `search_web` inside the normal agent loop
- Ollama tool-call parsing plus a bounded observation-action loop for native-tool profiles
- Guided onboarding that rewrites local config files and can optionally start Ollama or pull missing models
- Per-profile `keep_alive` settings sent to Ollama chat requests
- Real Ollama provider/runtime integration tests plus a `real-ollama-integration` GitHub Actions workflow
- Local logs and tracing

### Important current limits

- The default profile in `config/app.yaml` is still `main`, which uses `tool_mode: json_plan`. Broad model-driven tool use is therefore not the default everyday path yet.
- `/read`, `/ls`, and `/fetch` remain mainly slash-command-driven in normal user flows.
- Search is bounded and synchronous: DuckDuckGo HTML plus a small set of fetched public pages.
- Private and local network fetches are blocked by default.
- Memory is session-oriented deterministic summary plus recent history, not the richer long-term multi-store memory described in the project vision.
- There is no public GUI yet.
- There is no full file or OS automation suite yet.
- There is no shipped skills marketplace or plugin marketplace yet.
- There is no rich multi-store memory retrieval tool architecture yet.
- General document editing automation is not a shipped headline feature yet.

## Transparency: web search and local logs

### How grounded web search works today

- `search_web` sends the query to DuckDuckGo's HTML endpoint (`https://html.duckduckgo.com/html/`), parses result titles and snippets, deduplicates and ranks up to 20 initial results, then fetches a bounded set of public pages to extract evidence.
- Retrieval is synchronous and bounded: at most 30 fetched pages, crawl depth 2, at most 3 child links added per fetched page, and at most 12,000 extracted characters kept per fetched search page before synthesis.
- The current search path only works with public pages it can fetch directly. It does not use browser automation, JavaScript rendering, authenticated sessions, or private sites. Search quality therefore depends on DuckDuckGo HTML markup and on readable text being present in the fetched page response.

### Web safety boundaries

- Both `/search` and `/fetch` only support direct HTTP and HTTPS targets.
- By default the fetch policy blocks `localhost`, `.localhost`, metadata-style hosts, literal private or local IPs, and DNS resolutions that land on loopback, link-local, private, reserved, multicast, or unspecified addresses. Redirect targets are checked again under the same rules.
- `fetch_url_text` can be reconfigured locally with `security.tools.fetch.allow_private_networks: true` in `config/app.yaml`. The grounded `search_web` path stays public-web-only and does not use that override.

### What local logs and traces record today

- The tracer publishes runtime events in-process and, in the normal CLI and Telegram startup paths, persists them locally in two places: the SQLite `events` table in `data/app.db` and the JSONL runtime log at `data/logs/runtime.log`.
- Runtime trace payloads include timestamps, session IDs, route and model selections, tool names and arguments, success or failure state, durations, Telegram chat IDs for Telegram-only events, and reply or output lengths. This means local search queries, file paths, and fetched URLs can appear in local traces.
- On normal bootstrap/startup paths, Unclaw prunes runtime trace artifacts older than `logging.retention_days` from both the SQLite `events` table and `data/logs/runtime.log`. The default is 30 days; set `logging.retention_days: 0` to disable that automatic trace cleanup.
- By default reasoning text is not persisted. The tracer records `reasoning_length` only. If you set `logging.include_reasoning_text: true`, raw reasoning text is stored in the same local event payloads and becomes visible in `unclaw logs full`.
- `unclaw logs` reads the local JSONL runtime log, not the SQLite `events` table. If `logging.file_enabled` is off, that file stops updating even though local event publishing and local event persistence still exist.
- Successful grounded search turns also store a compact tool-history message in session history for follow-up grounding: supported facts, uncertain details, and source URLs rather than full fetched page dumps.

---

## What Unclaw is today

Unclaw is currently an **early but real local-first agent/runtime MVP**.

For ordinary turns it usually does one model call. For web-backed turns it can route the request into grounded search, persist the retrieved context, and answer with compact sources. On native-tool profiles, the same runtime can keep going through a bounded observation-action loop and let the model call built-in tools such as `search_web`.

The important limit is that this broader agent behavior is still selective rather than universal. The shipped `deep` profile is agent-capable, but the default `main` profile and most normal file/fetch flows remain conservative. Unclaw should therefore be described as a local-first AI agent/runtime that is still moving toward a more generally autonomous default experience.

---

## Quick start

Unclaw targets **Python 3.12+** and uses **Ollama** for local models.

```bash
 git clone https://github.com/nidrajud/unclaw.git
 cd unclaw
 python3.12 -m venv .venv
 source .venv/bin/activate
 python -m pip install -e .[dev]
 ollama serve
 unclaw onboard
 unclaw start
```

---

## Main commands

```bash
unclaw start
unclaw telegram
unclaw onboard
unclaw logs
unclaw logs full
unclaw update
unclaw help
```

### In-session commands

```bash
/help
/new
/sessions
/use <session_id>
/model
/model fast
/model main
/model deep
/model codex
/think on
/think off
/tools
/read <path>
/ls [path]
/fetch <url>
/search <query>
/session
/summary
```

---

## Current model profiles

- **fast** → quick lightweight replies, `tool_mode: json_plan`, `keep_alive: 10m`
- **main** → default everyday assistant, `tool_mode: json_plan`, `keep_alive: 30m`
- **deep** → heavier reasoning and the current shipped native-tool profile, `tool_mode: native`, `keep_alive: 10m`
- **codex** → code-oriented tasks, `tool_mode: json_plan`, `keep_alive: 10m`

Default lineup:

- `fast` → `llama3.2:3b`
- `main` → `qwen3.5:4b`
- `deep` → `qwen3.5:9b`
- `codex` → `qwen2.5-coder:7b`

The default onboarding recommendations match this lineup. If you want native tool calling in your everyday default path, switch the default profile to `deep` or configure another compatible profile with `tool_mode: native`.

---

## Tests and CI

The repository includes both mocked provider tests and live local Ollama coverage:

- unit and integration tests for the Ollama provider and shared runtime
- `tests/conftest.py` shared fixtures for the current test suite
- optional `@pytest.mark.real_ollama` tests for live local Ollama runs
- a `real-ollama-integration` GitHub Actions workflow that installs Ollama, pulls a repo-aligned model, and runs the live integration subset

---

## Security mindset

Unclaw follows one rule:

**safe defaults first, more power later by explicit choice.**

Current protections already include:

- deny-by-default Telegram access with `allowed_chat_ids: []` until a local allowlist entry is added
- Telegram token storage in local `config/secrets.yaml` or an env var, with masking in logs/errors
- owner-only `0o600` hardening for local secrets files and the SQLite database on POSIX systems
- allowed-root file access
- public HTTP/HTTPS-only web access by default, with private-network fetch blocking
- untrusted tool/search output wrappers plus instruction-like line tagging before tool results are fed back to the model
- reasoning text excluded from logs by default

---

## Long-term goal

Unclaw aims to become a **serious local AI assistant for everyday use**:

- search the web
- summarize information
- read and manipulate files
- help write code and text
- interact through multiple channels
- stay private, local, and under user control

If the project succeeds, Unclaw should feel like having a small personal assistant on your own machine — not renting one from the cloud.
