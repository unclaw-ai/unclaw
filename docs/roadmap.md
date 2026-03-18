# Roadmap

## Strategy

This document is the authoritative source for current shipped status and remaining roadmap work.

Unclaw should keep moving in layers, but the roadmap must stay honest about what already ships.
When another strategic document overlaps with current status, defer to this file.

Today the repository is no longer just a conversational assistant skeleton. It already includes:

- a shared local runtime
- model-assisted routing
- grounded web search
- a bounded native agent loop
- one shipped native-tool profile (`deep`)
- onboarding, tracing, and local persistence

The remaining work is to make that agent/runtime behavior broader, more default, and more polished without drifting into a heavyweight or misleading architecture.

---

## Current audit status snapshot

### Completed in the current repo

- `main` (default) and `deep` profiles both have `tool_mode: native`; `fast` and `codex` remain `json_plan`
- all profiles now have explicit `num_ctx` values (4096 or 8192) passed to Ollama to prevent silent context overflow
- the system prompt now includes tool-use, anti-injection, and citation guidance
- Ollama requests now send per-profile `keep_alive`
- normal routed web-backed turns and `/search` both use the shared runtime path
- native-tool profiles can call `search_web` inside the normal agent loop
- shared test fixtures now exist in `tests/conftest.py`
- real Ollama provider/runtime integration tests exist, and CI runs a live Ollama subset
- committed Telegram config is now deny-by-default with `allowed_chat_ids: []`
- local secret files and the SQLite database are hardened to owner-only permissions on POSIX
- tool and search output is wrapped as untrusted content before being fed back to the model
- `system_info` tool, notes tool family (`create_note`, `read_note`, `list_notes`, `update_note`), `write_text_file`, `inspect_session_history`, and long-term memory tools are now registered and active

### Still remaining

- make broader model-driven tool use the normal default path, not mainly `deep` and search
- expand routing beyond chat versus web-backed search
- ship richer multi-store memory and selective retrieval
- add a public GUI or other richer primary interface
- broaden safe automation without overselling current capabilities

---

## Phase 0 — Foundation and docs

### Goal
Keep the repository coherent and keep project positioning aligned with reality.

### Status
Core foundation is complete. Documentation alignment is ongoing and must continue whenever the runtime meaningfully changes.

### Already in place

- README and core strategic docs
- local Python package and CLI entrypoints
- local config structure
- local persistence and trace storage

### Remaining expectation

- keep docs aligned with the actual shipped runtime and security posture

---

## Phase 1 — Base local runtime

### Goal
Ship a useful local-first conversational runtime with shared infrastructure.

### Status
Complete.

### Delivered

- terminal CLI
- Telegram polling channel
- session management
- local persistence
- Ollama provider abstraction
- model profiles
- startup checks and onboarding
- logging and trace plumbing

---

## Phase 2 — Safe practical tools

### Goal
Make the runtime useful beyond plain chat.

### Status
Complete for the current MVP scope.

### Delivered

- grounded web search
- direct URL fetch
- file read
- directory listing
- tool registry and dispatcher
- modularized web-search stack
- traceable manual tool execution

### Remaining limitation

- outside search, these tools still lean heavily on explicit slash commands in normal use

---

## Phase 3 — Agent/runtime transition

### Goal
Turn the shared runtime into a more genuinely agentic default experience without losing safety or simplicity.

### Status
In progress.

### Landed already

- native tool definitions and tool-call parsing in the Ollama provider
- a bounded observation-action loop in the shared runtime
- `main` (default) and `deep` profiles both use native tool calling; `fast` and `codex` remain `json_plan`
- explicit `num_ctx` per profile prevents silent context overflow
- expanded system prompt for tool use and anti-injection behavior
- per-profile `keep_alive`
- model-assisted routing between chat and web-backed search
- shared `/search` handling through the same runtime path as normal turns
- native `search_web` calls inside the normal agent loop when a native profile is selected
- `system_info`, notes family, `write_text_file`, `inspect_session_history`, and long-term memory tools are now shipped

### Still missing before this phase can be called complete

- reduce reliance on `/read`, `/ls`, and `/fetch` as the primary access path for those capabilities
- broaden capability routing beyond the current chat versus web-search split
- make the default UX feel less command-heavy while keeping power-user commands

---

## Phase 4 — Stronger memory foundations

### Goal
Improve continuity without bloating prompts.

### Status
Partially complete.

### Landed already

- persisted session history in SQLite (`data/memory/app.db`)
- deterministic session summaries
- retained grounded facts and uncertainties from prior search results
- bounded session-memory context injection
- per-session JSONL chat mirror under `data/memory/chats/` (thread-safe access via `inspect_session_history` tool)
- long-term cross-session memory in `data/memory/long_term.db` with explicit recall tools; not injected automatically

### Remaining

- richer project memory
- selective retrieval across memory stores beyond the current explicit-call model
- clearer memory write and retention policies for longer-running sessions

---

## Phase 5 — Stabilization and quality hardening

### Goal
Make the current runtime more reliable, testable, and honest to operate.

### Status
Partially complete.

### Landed already

- shared test fixtures in `tests/conftest.py`
- real Ollama integration coverage and CI
- stronger prompt-injection handling around tool/search output
- deny-by-default Telegram allowlist configuration
- local secret and database permission hardening
- continued startup and onboarding polish

### Remaining

- continue filling critical runtime edge-case coverage where needed
- keep docs aligned as the runtime changes
- continue cleanup only where it directly improves shipped behavior or auditability

---

## Phase 6 — Research depth and caching

### Goal
Strengthen web and information workflows without bloating the runtime.

### Status
Not complete.

### Already in place

- grounded web search
- bounded retrieval
- follow-up grounding from prior search context

### Remaining

- local research caching
- deeper research/session state
- quick versus deep research profiles
- stronger bounded multi-step research workflows

---

## Phase 7 — More tools and safe automation

### Goal
Expand capability surface carefully.

### Status
Partially landed; most scope is still future work.

### Already landed

- `system_info` (read-only local machine info: OS, hostname, date/time, locale)
- notes tool family (`create_note`, `read_note`, `list_notes`, `update_note` in `data/notes/`)
- `write_text_file` (permissioned writes inside allowed roots; relative paths go to `data/files/`)

### Remaining scope

- browser automation
- draft generation
- code and project tools
- deeper local system actions
- stronger permission boundaries
- possible skill packages

### Constraint

Do not widen the tool surface faster than the orchestration quality, permission model, and observability can support.

---

## Phase 8 — Multi-machine local orchestration

### Goal
Use stronger local machines when available.

### Status
Future work.

### Scope

- wake-on-LAN
- local worker dispatch
- secure machine-to-machine coordination
- result retrieval for heavier jobs

---

## Phase 9 — Voice and richer UX

### Goal
Add richer interfaces once the runtime foundation is genuinely solid.

### Status
Future work.

### Scope

- local STT
- local TTS
- voice sessions
- richer UI
- execution timeline visualization

### Constraint

Do not use UI polish to hide architectural gaps.

---

## Immediate roadmap truth

The next important milestone is not “more features”.

It is:

- make the current agent/runtime behavior broader in the default experience
- keep search, memory, and security claims honest
- preserve the local-first, lightweight, and secure-by-default architecture while that expansion happens
