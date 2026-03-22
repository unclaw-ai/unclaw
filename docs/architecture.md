# Architecture

## Purpose

This document describes the **current shipped architecture** of Unclaw.

It focuses on how the runtime is structured today. For product philosophy and long-term direction, see [docs/vision.md](vision.md). For contributor constraints and priorities, see [docs/project_brief.md](project_brief.md).

---

## Current shipped posture

Today Unclaw ships as:

- a shared local runtime for terminal and Telegram
- local Ollama model profiles with `fast`, `main`, `deep`, and `codex`
- guided onboarding that writes local config files and optional local secrets
- model-assisted routing between normal chat and a web-backed search path
- grounded search as the most integrated model-driven tool path today
- deterministic session summaries injected as bounded context notes
- local SQLite persistence plus local JSONL and SQLite tracing
- a bounded native tool loop on profiles configured for native tool calling

Current profile posture:

- `main` (default) and `deep` use `tool_mode: native` — agent loop is active by default
- `fast` and `codex` use `tool_mode: json_plan`
- all profiles now set explicit `num_ctx` (4096 or 8192)

---

## Main runtime layers

### 1. Channels

Shipped channels are:

- terminal CLI via `unclaw start`
- Telegram polling bot via `unclaw telegram`

These layers are intentionally thin. They are responsible for:

- accepting user input
- running startup checks
- rendering replies
- handling channel-specific session binding or message formatting
- delegating actual routing, model calls, and tool orchestration to the shared runtime

### 2. Shared core runtime

The shared runtime lives in `src/unclaw/core/` and owns:

- route selection
- capability summaries for prompt honesty
- context assembly
- orchestration of model calls
- bounded observation-action loops
- explicit tool execution
- grounded search reply shaping
- persistence of assistant and tool history

### 3. LLM provider boundary

`src/unclaw/llm/ollama_provider.py` is the shipped provider boundary.

It is responsible for:

- sending chat requests to the local Ollama HTTP API
- forwarding `think` on/off state
- forwarding per-profile `keep_alive`
- forwarding native tool definitions when the selected profile supports them
- parsing streamed and non-streamed replies
- parsing native `tool_calls`
- stripping leaked `<think>` blocks from assistant-visible text

Provider-specific HTTP details stay here rather than leaking into channel code.

### 4. Tools

Tools are registered capability endpoints, not separate UX products.

Current built-in tools include:

- `search_web`
- `fetch_url_text`
- `read_text_file` (`.txt`, `.md`, `.json`, `.csv` only; binary formats not supported in V1)
- `list_directory`
- `write_text_file` (permissioned; fails on existing files unless `overwrite=true`; relative paths go to `data/files/`)
- `system_info` (read-only local machine snapshot)
- `inspect_session_history` (exact recall of persisted session messages)
- `remember_long_term_memory`, `search_long_term_memory`, `list_long_term_memory`, `forget_long_term_memory`

Tools return data. The runtime owns final answer composition.

### 5. Persistence and memory

The local persistence layer is SQLite plus flat files under `data/`.

Current persisted state includes:

- sessions and messages in `data/memory/app.db` (SQLite)
- runtime events in the same SQLite database and JSONL traces in `data/logs/runtime.log`
- deterministic session summaries stored in the session record
- per-session JSONL chat mirrors under `data/memory/chats/` (thread-safe history access)
- long-term cross-session memory in `data/memory/long_term.db` (SQLite)

Memory layers today:

- **Session layer**: recent conversation history (up to 20 messages) plus one deterministic session summary (no LLM call; extracts last few intents, grounded facts, uncertainties, and snippets)
- **Chat mirror**: append-only JSONL per session for `inspect_session_history` tool access from any thread
- **Long-term layer**: explicit cross-session facts and preferences in `long_term.db`; not injected automatically; the model retrieves via recall tools on explicit user request

The long-term memory layer is intentionally conservative: the model stores only when the user explicitly asks, and retrieves only when the user asks.

---

## Startup, settings, and onboarding flow

### Settings loading

`load_settings()` in `src/unclaw/settings.py` loads:

- `config/app.yaml`
- `config/models.yaml`
- `config/prompts/system.txt`

It validates and resolves:

- runtime paths
- channel toggles
- logging and retention settings
- tool security settings
- model profiles, including `tool_mode` and `keep_alive`

### Bootstrap

`bootstrap()` in `src/unclaw/bootstrap.py` prepares the local runtime by:

- creating required runtime directories
- applying log and trace retention pruning

When channels later open the SQLite database through the session manager, the database file is re-hardened to owner-only permissions on POSIX systems.

### Onboarding

`unclaw onboard` runs the interactive onboarding flow in `src/unclaw/onboarding.py`.

It currently:

- inspects the local Ollama installation
- loads existing Telegram config and local secrets
- recommends a default local model lineup
- writes `config/app.yaml`, `config/models.yaml`, and `config/telegram.yaml`
- optionally writes `config/secrets.yaml` for the Telegram token
- can guide the user to start Ollama or pull missing models

The shipped onboarding recommendations currently match the repo defaults:

- `main` is the default profile and uses native tool calling
- `deep` is the heavier-model native-tool profile, recommended for complex or multi-step tasks
- `fast` and `codex` remain `json_plan`

---

## Channel boundaries

### Terminal CLI

`src/unclaw/channels/cli.py` is a REPL wrapper around the shared runtime.

It handles:

- startup reporting
- slash commands
- streamed terminal rendering
- local display of tool results
- summary refresh after each turn

It does not contain its own routing or tool-selection policy.

### Telegram

`src/unclaw/channels/telegram_bot.py` is a polling-based channel that adds:

- deny-by-default chat authorization
- per-chat session binding
- per-chat command handlers
- message chunking
- per-chat worker isolation when the isolated-worker dependencies are available
- simple burst rate limiting

Telegram-specific management commands stay local to the machine running Unclaw. Remote chats cannot authorize themselves.

---

## Current shared runtime flow

For plain chat turns, the shared runtime flow is:

1. the channel persists the user message
2. `run_user_turn()` resolves tool availability for the selected model profile
3. the runtime builds a capability summary, optional session memory context note, and optional active-skill catalog
4. `Orchestrator.run_turn()` builds the actual LLM message list from:
   - the system prompt
   - runtime capability context
   - optional active-skill catalog (separate system message when skills are enabled)
   - recent session history
   - grounded-search answer contract when relevant
5. the Ollama provider is called with built-in tool definitions plus any active skill tool definitions
6. the runtime either returns the model text directly or enters the bounded agent loop
7. the final assistant reply is transformed if needed, persisted, and traced

There is no separate routing model or classifier step. The main model decides directly whether to answer, call a tool, or invoke a skill.

---

## Native agent loop

The native agent loop is implemented in `src/unclaw/core/runtime.py`.

It activates only when the selected profile supports native tool calling.

Current behavior:

- native tool definitions are sent only for native-tool profiles
- the loop is bounded by a step limit and a tool-call budget
- tool calls are executed concurrently within one batch
- tool calls respect a runtime timeout
- tool results are persisted to session history
- tool results are wrapped as untrusted content before being fed back to the model
- the loop stops when the model returns a final text reply or a bound is hit

This is a real observation-action loop, not a simulated slash-command wrapper.

Important limit:

- the default `main` profile uses `native` tool mode, so the agent loop is the default everyday experience

---

## Tool execution path

There are two current tool execution modes.

### Explicit slash-command path

For `/read`, `/ls`, `/fetch`, and `/search`:

- `CommandHandler` resolves the explicit command
- the channel executes the tool or hands `/search` to the shared research flow
- the result is persisted as tool history
- the user sees the tool result or the grounded assistant reply

### Native model-callable path

For native profiles:

- Ollama can return `tool_calls`
- the runtime executes them through the shared tool registry and dispatcher
- tool results are persisted in model order
- the model receives tool results back as untrusted tool messages
- the loop continues until a final answer is ready

---

## Search grounding path

Grounded search is part of the shared runtime, not a separate assistant product.

Current behavior:

- `/search` goes through `src/unclaw/core/research_flow.py`
- normal turns can also be routed into a web-backed search path by the shared router
- on non-native profiles, the runtime pre-executes `search_web` before the final model answer
- on native profiles, the model can call `search_web` inside the normal agent loop
- search tool results are persisted as compact tool-history summaries
- the final reply is shaped from the latest grounded search context and then given compact sources

The underlying search stack is still:

- DuckDuckGo HTML discovery
- bounded public page retrieval
- grounding and source shaping

Important current limits:

- retrieval is synchronous
- search is public-web-only
- search still depends on DuckDuckGo HTML and readable fetched page text

---

## Memory and context model

Current memory behavior is deliberately modest.

`MemoryManager` builds a deterministic session summary from persisted messages and search tool history. That summary can inject a single bounded context note that includes:

- recent user intents
- retained grounded facts
- retained uncertainties
- the latest assistant reply summary
- basic session size metadata

This gives follow-up turns useful continuity without pretending Unclaw already has:

- user memory
- project memory
- execution memory stores
- retrieval-augmented long-term memory selection

---

## Security and trust boundaries

### Untrusted tool and search content

Tool output is not injected back into the model as plain trusted text.

`build_untrusted_tool_message_content()` in `src/unclaw/core/context_builder.py`:

- wraps tool output in an explicit untrusted-content block
- warns that trusted instructions come only from runtime/system messages
- flags instruction-like lines before the model sees them again

This is a meaningful hardening layer, but it does not fully eliminate prompt-injection risk from weak local models.

### Web safety

Current web boundaries are conservative:

- `search_web` stays on the public web path
- fetch/search require direct HTTP or HTTPS targets
- private-network and local-address SSRF targets are blocked by default
- redirect targets are revalidated

### File safety

Local file tools are restricted to configured allowed roots.

### Telegram safety

Telegram is deny-by-default:

- `config/telegram.yaml` ships with `allowed_chat_ids: []`
- unauthorized chats are rejected and logged locally
- allow/revoke/list management requires local CLI access

### Local secrets and local data

The current local security posture also includes:

- Telegram token storage in `config/secrets.yaml` or an environment variable
- token masking in logs and Telegram API errors
- owner-only `0o600` hardening for local secrets files on POSIX
- owner-only `0o600` hardening for the SQLite database on POSIX
- reasoning text excluded from logs by default
- automatic local trace pruning based on `logging.retention_days`

---

## Current limitations that remain intentional to document

These are current runtime boundaries, not hidden roadmap promises:

- no public GUI
- no general local API layer documented as a shipped primary interface
- no broad default model-driven file or fetch automation path
- no rich multi-store memory retrieval architecture
- no skills marketplace
- no broad OS automation suite
- no finished document-editing automation workflow marketed as a core feature
