# Skills Design

## Status

This document defines the target architecture for optional skills in Unclaw.
It is a design reset point for Phase 1 only.

This document does not change the shipped runtime yet.
It does not migrate existing skills yet.
It does not redesign tools, routing, memory, or orchestration.

## Why this reset is needed

The current optional-skill path is heavier than it should be.
Today skill prompt content is modeled through typed Python manifests in:

- `src/unclaw/skills/models.py`
- `src/unclaw/skills/manifests.py`
- `src/unclaw/skills/registry.py`
- `src/unclaw/skills/runtime.py`

That structure works, but it adds Python metadata, registry plumbing, prompt-fragment bookkeeping, and validation machinery for something that should stay lightweight.

At the same time, Unclaw already has a clear native tool boundary:

- real execution lives in Python tool handlers
- the runtime exposes tools through `ToolDefinition`, `ToolRegistry`, and `ToolDispatcher`
- prompt honesty is built around compact capability summaries

Skills keeps that strong tool boundary and simplifies optional skills into file-first prompt bundles.

## Core definitions

### Native built-in tools

A native built-in tool in Unclaw is a real runtime capability implemented in Python.

Properties:

- execution happens in Python, not in prompt text
- the tool has a typed contract and a real handler
- the tool owns validation, I/O, network access, permissions, and payload shape
- the runtime may expose it directly to models through the normal tool path
- the runtime remains responsible for truthfulness, tracing, persistence, and final answer composition

Examples in the current repo:

- file tools in `src/unclaw/tools/file_tools.py`
- terminal execution in `src/unclaw/tools/terminal_tools.py`
- system info in `src/unclaw/tools/system_tools.py`
- memory tools in `src/unclaw/tools/long_term_memory_tools.py`
- weather lookup, which is currently mixed but should ultimately behave like a normal built-in tool

### Optional skills

An optional skill in Unclaw is prompt-level guidance packaged as a lightweight bundle.

Properties:

- a skill is centered on a `SKILL.md`
- a skill does not execute anything by itself
- a skill does not own runtime permissions
- a skill does not replace built-in tools
- a skill packages workflow guidance, usage rules, and task-specific instructions around existing runtime capabilities
- a skill may include optional supporting files later, such as references, examples, or templates

Skills are therefore guidance bundles, not runtime subsystems.

## Boundary: Python code vs `SKILL.md`

### Belongs in Python

- tool definitions and handlers
- tool registration
- argument validation and coercion
- filesystem, shell, network, and database access
- typed payload contracts
- safety guards that enforce real behavior
- tracing, persistence, and tool-result grounding
- any code that can succeed, fail, mutate state, or touch the outside world

### Belongs in `SKILL.md`

- when the skill is relevant
- which built-in tools it expects the model to use
- workflow guidance for that domain
- domain-specific answer-shaping rules
- domain-specific truthfulness reminders
- examples, checklists, and lightweight references

Rule of thumb:

- if it executes, validates, or enforces, it belongs in Python
- if it teaches, guides, scopes, or packages a workflow, it belongs in `SKILL.md`

## Proposed skill bundle shape

Shipped optional skills should move to simple directories under the repo-root `skills/`.

Example:

```text
skills/
  weather/
    SKILL.md
    references/
    examples/
```

`SKILL.md` is required.
Other files are optional.

The bundle directory is the source of truth for the skill.
Phase 2 should treat current manifest-based skill modules as transitional and move away from them.

The new skill id should be the bundle directory name, using simple slugs such as `weather`.
Current manifest ids like `information.weather` can be supported as migration aliases temporarily, but they should not remain the long-term source of truth.

## Discovery and activation

### Discovery

Phase 2 should discover shipped skills by scanning `skills/` for directories that contain `SKILL.md`.

Discovery should be:

- deterministic
- local-only
- filesystem-based
- cheap enough to cache in memory for the process lifetime

No Python manifest import should be required for discovery.

### Activation

For the first new implementation, Unclaw should keep the existing config concept of explicitly enabled skills in `config/app.yaml`.

That means:

- the runtime discovers all shipped skill bundles
- `skills.enabled_skill_ids` selects which ones are active for this installation
- activation remains opt-in by config, not automatic marketplace behavior

Phase 2 may support temporary aliases so existing config values such as `information.weather` continue to resolve while the repo migrates to bundle-based ids.

## Prompt strategy

### What is injected by default

By default, the runtime should inject only a compact catalog of active skills, not the full contents of every skill.

That catalog should stay short and include only the minimum needed to help the model notice relevant skills, for example:

- skill name
- one-line purpose
- key built-in tools it expects

Example shape:

```text
Active optional skills:
- weather: Live weather and short forecasts. Prefer get_weather; use search_web only as fallback for official alerts or missing details.
```

This keeps prompt cost low on local machines and matches Unclaw's existing capability-budget discipline.

### When a full skill is loaded

The full `SKILL.md` for a skill should be loaded only when that specific skill is relevant to the current turn.

Relevant means one of:

- the user explicitly asks for the skill by name
- the user request clearly falls into the skill's scoped domain
- the runtime selects the skill from the compact catalog using a simple local matching step

The exact matching heuristic is a Phase 2 implementation detail, but it must remain small, deterministic, and cheap.
This phase does not require a new planner or router.

Default guardrails:

- load at most the selected skill's `SKILL.md` for a normal turn
- only load more than one skill if a later use case proves it is necessary
- never inline full content for every active skill by default

## Weather in the new model

Weather remains the first migration candidate and stays part of the optional skill model.

Target split:

- `get_weather` is a native built-in tool implemented and registered like other serious Python tools
- the weather skill is a `SKILL.md` bundle that teaches when and how to use `get_weather`
- `search_web` remains a fallback tool for alerts or details outside the dedicated weather backend

Why weather goes first:

- it already has a useful grounded tool contract
- it already has a clear optional-guidance story
- it has integration tests that prove the current behavior is valuable
- it is the cleanest example of "tool in Python, workflow in skill text"

Current mixed ownership should be treated as transitional:

- execution lives in `src/unclaw/skills/weather/tool.py`
- the compatibility export lives in `src/unclaw/tools/weather_tools.py`
- prompt guidance lives in `src/unclaw/skills/manifests.py`

Phase 2 should simplify that into one built-in weather tool path plus one file-first weather skill bundle.

## Sticky notes are out of scope

Sticky notes are explicitly paused for this redesign.

Reason:

- Unclaw intentionally has no shipped built-in notes subsystem right now
- future document creation and sticky/post-it behavior are separate product surfaces
- "sticky notes" or desktop post-it behavior is a product-surface question, not just a skill-format question
- the boundary between document creation, desktop UI behavior, and any future skill wrapper is not clear enough yet
- the current repo does not have a stable checked-in sticky-notes implementation to preserve

Therefore:

- do not continue sticky-notes work in the current skill architecture
- do not use sticky notes to justify new runtime plumbing
- do not reintroduce the removed legacy notes subsystem through a skill wrapper
- revisit this only after the product boundary is redesigned clearly

## Migration path from the current repo

Phase 2 should implement this in small steps:

1. Add a file-first skill loader that discovers `SKILL.md` bundles and can build a compact active-skill catalog.
2. Keep `skills.enabled_skill_ids` as the activation switch initially, with temporary alias support for current dotted ids if needed.
3. Add on-demand full-skill loading for a selected skill without changing the core tool loop or memory architecture.
4. Migrate the weather skill text out of manifest fragments into `skills/weather/SKILL.md`.
5. Normalize weather execution ownership so `get_weather` is clearly a built-in Python tool, not a skill-owned runtime subsystem.
6. Once the weather path works, migrate other optional skills away from Python manifests.
7. Remove the old manifest/registry/prompt-fragment skill machinery after parity is reached.

## Explicit non-goals for Phase 2

Phase 2 should not use this redesign as a reason to:

- rewrite the runtime
- redesign tool calling
- redesign the orchestrator
- redesign memory
- build a skill marketplace or installer
- continue sticky-notes feature work in its current form
- migrate every existing skill at once

## Decision summary

Skills resets Unclaw to a simpler rule set:

- tools are Python
- skills are files
- the prompt gets a compact skill catalog by default
- full skill text is loaded only on demand
- weather is the first migration target
- sticky notes are paused until their product boundary is clear

This is the right reset point because it preserves Unclaw's strongest current qualities:

- honest local tool execution
- lightweight prompting
- explicit runtime boundaries
- good performance on small local machines

It also gives Phase 2 a narrow, practical implementation target instead of another round of architecture growth.
