<p align="center">
  <img src="./unclaw-animated.svg" alt="Unclaw animated logo" />
</p>

<h1 align="center">Unclaw 🦐</h1>

<p align="center"><strong>Local-first AI runtime for a real personal assistant.</strong></p>

<p align="center">No cloud. No lock-in. No unnecessary framework bloat.</p>

Unclaw is a lightweight runtime for local AI models.

Today it is a local-first assistant runtime with terminal and Telegram access, local tools, session persistence, grounded web search, and local logs.

It also contains the plumbing for a bounded tool-calling loop, but the shipped profiles still keep tool use conservative and mostly command-driven by default.

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
- Interactive terminal via `unclaw start`
- Telegram polling bot with local allowlist management
- Session persistence plus deterministic session summaries
- Slash commands for `/read`, `/ls`, `/fetch`, and grounded `/search`
- Model-assisted routing between normal chat and web-backed search
- Grounded search replies with compact sources and grounded follow-up turns
- Ollama tool-call parsing and a bounded observation-action loop for profiles configured with `tool_mode: native`
- Guided onboarding that rewrites local config files and can optionally start Ollama or pull missing models
- Local logs and tracing

### Important current limits

- The shipped profiles in `config/models.yaml` and the onboarding-recommended lineup all use `tool_mode: json_plan`, so native model-callable tools are not enabled by default.
- `/read`, `/ls`, and `/fetch` remain slash-command-driven in normal user flows.
- Search is bounded and synchronous: DuckDuckGo HTML plus a small set of fetched public pages.
- Private and local network fetches are blocked by default.
- Memory is session-oriented summaries, not rich user/project memory.

---

## What Unclaw is today

Unclaw is currently an **early but real MVP**.

For ordinary turns it usually does one model call. For web-backed turns it can route the request into grounded search, persist the retrieved context, and answer with compact sources. When a profile is configured with `tool_mode: native`, the runtime can continue through a bounded observation-action loop.

The important limit is that this native tool loop is **not** the default shipped setup yet, so Unclaw should still be described as a local assistant runtime moving toward broader agent behavior, not as a finished autonomous agent.

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

- **fast** → quick lightweight replies
- **main** → default everyday assistant
- **deep** → heavier reasoning
- **codex** → code-oriented tasks

Default lineup:

- `fast` → `llama3.2:3b`
- `main` → `qwen3.5:4b`
- `deep` → `qwen3.5:9b`
- `codex` → `qwen2.5-coder:7b`

All four shipped profiles currently use `tool_mode: json_plan`. If you want to exercise native tool calling, you need to switch a profile to `tool_mode: native` and use a compatible Ollama model.

---

## Security mindset

Unclaw follows one rule:

**safe defaults first, more power later by explicit choice.**

Current protections already include deny-by-default Telegram access, local secrets handling, allowed-root file access, public HTTP/HTTPS-only web access by default, private-network fetch blocking, and reasoning text excluded from logs by default.

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
