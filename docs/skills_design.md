# Skills Design

## Status

Migration complete. The file-first skill architecture described in this document
is now the only skill system in Unclaw. The legacy manifest-based system
(`models.py`, `manifests.py`, `registry.py`, `runtime.py`) has been removed.

This document is preserved as architectural reference.

## Why this reset was needed

The previous optional-skill path was heavier than needed.
Skill prompt content was modeled through typed Python manifests, adding registry
plumbing, prompt-fragment bookkeeping, and validation machinery for something
that should stay lightweight.

Unclaw already had a clear native tool boundary:

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

The skill id is the bundle directory name, using simple slugs such as `weather`.

## Discovery and activation

### Discovery

Unclaw discovers shipped skills by scanning `skills/` for directories that contain `SKILL.md`.

Discovery is:

- deterministic
- local-only
- filesystem-based
- cheap enough to cache in memory for the process lifetime

No Python manifest import is required for discovery.

### Activation

Activation uses the existing config concept of explicitly enabled skills in `config/app.yaml`.

- the runtime discovers all shipped skill bundles
- `skills.enabled_skill_ids` selects which ones are active for this installation
- activation is opt-in by config, not automatic marketplace behavior

## Prompt strategy

### What is injected by default

By default, the runtime injects only a compact catalog of active skills, not the full contents of every skill.

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

The full `SKILL.md` for a skill is loaded only when that specific skill is relevant to the current turn.

Relevant means one of:

- the user explicitly asks for the skill by name
- the user request clearly falls into the skill's scoped domain
- the runtime selects the skill from the compact catalog using a simple local matching step

The matching heuristic is small, deterministic, and cheap (see `selector.py`).
No planner or router changes are needed.

Default guardrails:

- load at most the selected skill's `SKILL.md` for a normal turn
- only load more than one skill if a later use case proves it is necessary
- never inline full content for every active skill by default

## Weather in the file-first model

Weather is the first migrated skill and illustrates the clean split:

- `get_weather` is a native built-in tool in `skills/weather/tool.py`, registered via the `register_skill_tools` hook
- the weather skill is a `SKILL.md` bundle that teaches when and how to use `get_weather`
- `search_web` remains a fallback tool for alerts or details outside the dedicated weather backend

This is the canonical example of "tool in Python, workflow in skill text".

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

## Completed migration steps

1. File-first skill loader discovers `SKILL.md` bundles and builds a compact active-skill catalog. ✓
2. `skills.enabled_skill_ids` is the activation switch; dotted-id aliases have been removed. ✓
3. On-demand full-skill loading is implemented for a selected skill without changing the core tool loop. ✓
4. Weather skill text lives in `skills/weather/SKILL.md`. ✓
5. Weather execution is a bundle-owned Python tool registered via `register_skill_tools`. ✓
6. Old manifest/registry/prompt-fragment skill machinery has been removed. ✓

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
