<p align="center">
  <img src="./unclaw-animated.svg" alt="Unclaw animated logo" />
</p>

<h1 align="center">Unclaw 🦐</h1>

<p align="center"><strong>Local-first AI agent/runtime for real local models.</strong></p>

<p align="center">No cloud. No lock-in. No unnecessary framework bloat.</p>

Unclaw is a lightweight local-first runtime for AI models served through Ollama.

Today it ships as a shared local runtime with terminal and Telegram channels, safe local tools, grounded web search, session persistence, onboarding, and local logs/traces.

It also ships a bounded agent loop. Both the default `main` profile and the `deep` profile enable native tool calling; `fast` and `codex` remain conservative `tool_mode: none` profiles.

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
- `main` (default) and `deep` profiles are both configured for native tool calling; `fast` and `codex` remain `tool_mode: none`; all profiles now have explicit `num_ctx` values (4096 or 8192) to prevent silent context overflow
- Interactive terminal via `unclaw start`
- Telegram polling bot with local allowlist management
- Session persistence plus deterministic session summaries
- Slash commands for `/read`, `/ls`, `/fetch`, and grounded `/search`
- Model-assisted routing between normal chat and web-backed search
- Grounded search replies with compact sources and grounded follow-up turns
- `search_web` integrated into the shared runtime path: non-native profiles use a runtime pre-executed search step, while native profiles can call `search_web` inside the normal agent loop
- Ollama tool-call parsing plus a bounded observation-action loop for native-tool profiles
- `system_info` built-in tool: read-only local machine snapshot (OS, Python version, CPU cores, RAM, hostname, date/time, locale)
- `write_text_file` for permissioned writes inside allowed roots; relative paths redirect to `data/files/` by default; fails on existing files unless `overwrite=true` is explicitly set
- `inspect_session_history` for exact recall of the current session's persisted messages (supports role filter, nth-message lookup, limit)
- Long-term cross-session memory: `remember_long_term_memory`, `search_long_term_memory`, `list_long_term_memory`, `forget_long_term_memory` — local SQLite at `data/memory/long_term.db`; not injected automatically; model retrieves on explicit request
- Per-session JSONL chat mirror under `data/memory/chats/` for thread-safe session history access
- `/memory-status` command for diagnostics on all active memory layers
- Guided onboarding that rewrites local config files and can optionally start Ollama or pull missing models
- Per-profile `keep_alive` settings sent to Ollama chat requests
- Real Ollama provider/runtime integration tests plus a `real-ollama-integration` GitHub Actions workflow
- Local logs and tracing

### Important current limits

- `fast` and `codex` profiles remain `tool_mode: none`; on those profiles the model is not offered native tool definitions in the turn, so the agent loop activates only on `main` and `deep`.
- `/read`, `/ls`, and `/fetch` remain mainly slash-command-driven in normal user flows.
- Search is bounded and synchronous: DuckDuckGo HTML plus a small set of fetched public pages. Search quality depends on DuckDuckGo HTML stability and readable page text.
- Private and local network fetches are blocked by default.
- File reading supports only `.txt`, `.md`, `.json`, and `.csv`; binary formats (pdf, docx, xlsx) are not supported in V1.
- There is intentionally no shipped built-in notes or document-creation tool right now.
- Long-term memory is not injected automatically — the model must call the recall tools explicitly; it is not yet a passive always-available context.
- Session summaries are deterministic (no LLM call): reliable and fast but limited in depth for complex multi-turn sessions.
- Lighter local profiles (`fast`, `main`) are less reliable on long multi-step tool chains than `deep`; `deep` is more capable for complex agentic tasks.
- There is no public GUI yet.
- There is no full file or OS automation suite yet.
- There is no shipped skills marketplace or plugin marketplace yet.
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

### How terminal execution works today

- `run_terminal_command` executes a real host command with Python `subprocess.run(..., shell=True, ...)`.
- The runtime validates and restricts the working directory to allowed roots or the configured default local working directory, but the command string itself is not sandboxed beyond the configured timeout and output limits.
- In practice this is real local shell execution on your machine, not a simulated terminal.

### What local logs and traces record today

- The tracer publishes runtime events in-process and, in the normal CLI and Telegram startup paths, persists them locally in two places: the SQLite `events` table in `data/app.db` and the JSONL runtime log at `data/logs/runtime.log`.
- Runtime trace payloads include timestamps, session IDs, route and model selections, tool names and arguments, success or failure state, durations, Telegram chat IDs for Telegram-only events, and reply or output lengths. This means local search queries, file paths, fetched URLs, and terminal commands can appear in local traces.
- On normal bootstrap/startup paths, Unclaw prunes runtime trace artifacts older than `logging.retention_days` from both the SQLite `events` table and `data/logs/runtime.log`. The default is 30 days; set `logging.retention_days: 0` to disable that automatic trace cleanup.
- By default reasoning text is not persisted. The tracer records `reasoning_length` only. If you set `logging.include_reasoning_text: true`, raw reasoning text is stored in the same local event payloads and becomes visible in `unclaw logs full`.
- `unclaw logs` reads the local JSONL runtime log, not the SQLite `events` table. If `logging.file_enabled` is off, that file stops updating even though local event publishing and local event persistence still exist.
- Successful grounded search turns also store a compact tool-history message in session history for follow-up grounding: supported facts, uncertain details, and source URLs rather than full fetched page dumps.
- When a streamed reply is rewritten by grounding after streaming begins, the CLI prints `[answer refined]` and the final grounded reply. This indicates the initial stream was superseded by a grounded answer.

---

## What Unclaw is today

Unclaw is currently an **early but real local-first agent/runtime MVP**.

For ordinary turns it usually does one model call. For web-backed turns it can route the request into grounded search, persist the retrieved context, and answer with compact sources. On native-tool profiles, the same runtime can keep going through a bounded observation-action loop and let the model call built-in tools such as `search_web`.

The `main` (default) and `deep` profiles are both native-tool-capable and can run the observation-action loop. The `fast` and `codex` profiles remain `tool_mode: none`, which means the model itself is not offered native tool definitions on those profiles. Local model reliability varies by task complexity — lighter profiles work well for focused tasks but are less reliable on long multi-step chains. The `deep` profile is the recommended choice for complex or multi-step agentic tasks.

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

Launch Unclaw from a dedicated workspace or project directory. By default,
`security.tools.files.allowed_roots: ["."]` is resolved from the
project/workspace root Unclaw discovers from your current working directory at
startup. If you run Unclaw from `~/` or keep the project rooted there, `/read`,
`/ls`, and `write_text_file` can inherit a much larger local file scope than
intended. The safest setup is to replace `"."` with explicit allowed root paths
in `config/app.yaml`.

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
/memory-status
```

---

## Current model profiles

- **fast** → quick lightweight replies, `tool_mode: none`, `num_ctx: 4096`, `keep_alive: 10m`
- **main** → default everyday assistant, `tool_mode: native`, `num_ctx: 8192`, `keep_alive: 30m`
- **deep** → heavier reasoning, native-tool profile, `tool_mode: native`, `num_ctx: 8192`, `keep_alive: 10m`
- **codex** → code-oriented tasks, `tool_mode: none`, `num_ctx: 4096`, `keep_alive: 10m`

Default lineup:

- `fast` → `ministral-3:8b`
- `main` → `ministral-3:14b`
- `deep` → `qwen3.5:27b`
- `codex` → `deepcoder:14b`

The default onboarding recommendations match this lineup. `main` is the default profile and runs native tool calling. Use `deep` for heavier reasoning or tasks that benefit from a larger model. `fast` and `codex` remain `tool_mode: none`, so they do not activate the native agent loop.

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
