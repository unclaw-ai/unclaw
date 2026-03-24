# No Deterministic Core Policy

## Purpose

This policy exists to stop Unclaw from drifting into a rules engine that only appears agentic while preserving the deterministic controls that are required for safe local execution.

## Definitions

### Agentic core

The agentic core is the part of the runtime where the model interprets the user's request, decides what information is needed, selects tools, adapts across steps, handles ambiguity, and produces the final answer.

### Deterministic scaffolding

Deterministic scaffolding is code-defined behavior that does not rely on model reasoning. It includes fixed routing rules, keyword triggers, hardcoded plan trees, language-specific hacks, and other logic that predetermines what the runtime will think or do.

### Safety guardrails

Safety guardrails are deterministic controls that restrict unsafe execution without substituting for reasoning. They include permission boundaries, path restrictions, SSRF protections, destructive-action confirmations, and similar execution-safety limits.

### Execution validation

Execution validation is deterministic checking that makes tool use safe and well-formed, such as schema validation, argument normalization, path resolution, URL checks, timeout enforcement, and budget limits.

### Multilingual safety

Multilingual safety means the runtime can operate without depending on FR-only or EN-only core logic for normal understanding, routing, planning, or language choice. Safety controls must remain effective without narrowing the agentic core to one language pair.

## Absolute Rule

Deterministic logic must not replace model reasoning in the agentic core.

## Forbidden Deterministic Patterns

The following patterns are forbidden in the agentic core:

- keyword or regex routing that decides which tool the model should call first
- language-specific routing logic for normal user intent
- hardcoded entity-disambiguation behavior that overrides model reasoning
- deterministic language selection for normal assistant behavior
- workflow trees that predetermine multi-step plans
- deterministic "if user says X, always do Y" behavior in the reasoning core
- FR-only or EN-only hacks introduced as temporary core behavior
- growing heuristic note systems that effectively script the model's reasoning path

## Allowed Deterministic Components

The following deterministic components are allowed when they exist to protect execution safety rather than replace reasoning:

- filesystem path restrictions
- permission enforcement
- SSRF and network safety controls
- command execution boundaries
- destructive-action confirmation flags
- schema validation
- input and output normalization that protects execution integrity
- tool registry boundaries
- rate and budget guards when they are execution-safety controls rather than reasoning substitutes
- timeout, size, and resource bounds on tools and loops

## Migration Rule for Existing Legacy Scaffolding

If deterministic scaffolding already exists, contributors must treat it as temporary legacy behavior. They must contain it, document it, test it, and replace it progressively with better model-side guidance, cleaner tool design, safer prompts, or stronger capability context. Legacy scaffolding must not be expanded into new core architecture.

## Multilingual Rule

The runtime must never depend on FR-only or EN-only core logic to understand the user. If a fallback is temporarily necessary, it must be language-neutral when possible and otherwise explicitly temporary, isolated, documented, and scheduled for removal. Universal behavior is required.

## Contributor Review Checklist

- Does this change replace model reasoning with code?
- Does it add language-specific behavior in the core?
- Is it safety logic or hidden orchestration?
- Can the same goal be achieved by better prompting, tool design, capability context, or model-side guidance instead?
- Does it scale across languages and future models?

## Acceptable vs Unacceptable Examples

- Acceptable: reject `read_text_file` outside allowed roots. Unacceptable: regex-route every "open file" request directly to `read_text_file` before model reasoning.
- Acceptable: block private or local-network fetch targets and revalidate redirects. Unacceptable: force all requests containing "latest" or "news" into `search_web` without the model deciding.
- Acceptable: validate tool arguments against a schema and reject malformed payloads. Unacceptable: hardcode "if the user asks for a summary, run tool A then tool B then answer."
- Acceptable: require explicit confirmation before a destructive write, delete, or shell action. Unacceptable: prebuild a deterministic research workflow tree and make the model fill in only the wording.
- Acceptable: normalize paths and URLs before permission and safety checks. Unacceptable: choose French or English response behavior through token heuristics in the normal reasoning path.
- Acceptable: cap tool-call count and step count to keep execution bounded. Unacceptable: hardcode person or entity disambiguation rules that override the model's interpretation of who or what the user means.

## Enforcement Rule

Any PR that adds deterministic reasoning logic to the agentic core should be rejected unless explicitly approved by the project owner.

## Final Policy Statement

If code replaces reasoning with rules, it does not belong in Unclaw's core.
