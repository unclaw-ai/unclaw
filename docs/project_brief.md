# Unclaw Project Brief

## 1. Purpose

This document is the concise practical brief for contributors and AI coding agents.

Use:
- [docs/vision.md](vision.md) for product philosophy and long-term direction,
- [docs/architecture.md](architecture.md) for the current runtime shape,
- [docs/roadmap.md](roadmap.md) for the authoritative shipped-status snapshot and remaining work.

---

## 2. Current project identity

**Name:** Unclaw
**Tagline:** Local-first AI runtime. No cloud lock-in. No recurring fees. Built for real local models.
**Emoji identity:** 🦐

Unclaw currently ships as a local-first assistant/runtime with:
- a shared core runtime for terminal and Telegram,
- local Ollama profiles (`fast`, `main`, `deep`, `codex`),
- grounded web search plus safe local file and fetch tools,
- session persistence, deterministic session summaries, and local traces,
- and a bounded native tool loop on compatible profiles.

Current reality contributors must keep in mind:
- the `main` (default) and `deep` profiles are both native-tool capable with explicit `num_ctx`,
- the `fast` and `codex` profiles remain `tool_mode: json_plan` and do not activate the agent loop,
- grounded search is the most integrated model-driven tool path today,
- `/read`, `/ls`, and `/fetch` are still mainly slash-command-driven in normal use,
- local tools (system_info, write_text_file, session history, long-term memory) are now shipped but the agent loop's reliability scales with model size,
- and the product is still moving toward a more broadly autonomous default experience — the foundation is in place but task complexity limits remain for small models.

Assume technical users on consumer hardware and keep claims honest about those current limits.

---

## 3. Non-negotiable constraints

Unclaw must remain:
- **local-first**,
- **privacy-first**,
- **secure-first**,
- **lightweight**,
- **open source and free to run locally**,
- and **architecturally honest**.

Practical meaning:
- no required cloud inference or cloud control plane,
- no heavy framework adoption without a clear need,
- no fake agent claims that hide manual or route-specific behavior,
- no brittle keyword lists as the main control plane,
- no erosion of explicit security boundaries around files, fetch, or tool output,
- and no removal of manual slash commands as power-user fallbacks.

---

## 4. Implementation guardrails

Unclaw should be extended as a runtime, not as a chat wrapper.

Contributors should preserve these boundaries:
- keep channels thin; routing, orchestration, and tool policy belong in the shared runtime,
- keep tool contracts explicit, validated, bounded, and traceable,
- keep the runtime responsible for the final natural answer while tools return data,
- prefer model-assisted or capability-based decisions over hand-maintained trigger lists,
- preserve bounded execution, untrusted-tool handling, and local observability,
- and keep memory claims modest; today the shipped memory layer is session history plus deterministic session summaries.

---

## 5. Current contributor priorities

When making changes, optimize for:
- broader model-driven tool use without overstating autonomy,
- docs and product claims that match shipped behavior,
- security and prompt-injection resilience around tool and search output,
- modular, auditable changes that do not add framework weight,
- and logs/traces that keep runtime behavior inspectable.

---

## 6. Current non-goals

These are not the focus of the current repo state:
- cloud-first product direction,
- framework-heavy rewrites,
- speculative GUI-first redesigns,
- unrestricted shell or OS automation,
- rich long-term memory marketed before it exists,
- marketplace or plugin positioning as a near-term identity,
- and broad product promises beyond the shipped runtime, search, and bounded native tool loop.

---

## 7. Working rules for contributors and AI coding assistants

- read the relevant audit material and current authority docs before changing behavior,
- keep patches narrow, reversible, and easy to audit,
- do not move shared-runtime logic into channels,
- do not introduce hidden heuristics or language-specific routing tables,
- do not describe partial capabilities as default or complete,
- and prefer brief cross-references over duplicating long explanations from the vision or architecture docs.
