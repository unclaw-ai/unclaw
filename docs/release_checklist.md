# Release Checklist

Use this before cutting or sharing a release build of the current runtime foundation.

## Scope

This checklist is for regression hardening of the shipped baseline. It is not a redesign gate.

## Required checks

- Run the targeted regression suite for the control surface, settings persistence, and startup reporting.
- Confirm `unclaw start` reaches preflight and shows:
  - startup/preflight status,
  - control preset summary,
  - default profile and context summary,
  - required model availability.
- Confirm `/control safe`, `/control workspace`, and `/control full` each:
  - save cleanly,
  - explain the new access level in plain language,
  - apply immediately in the current CLI.
- Confirm `/ctx` persists the requested override and reports effective behavior explicitly:
  - active profile: Unclaw attempts an immediate model refresh,
  - refresh success: next turn in the same CLI uses the new context,
  - refresh failure: CLI says the new value is guaranteed on next model reload or CLI restart.
- Confirm `/profiles` still lists each profile, model, tool mode, and context window.
- Confirm one `system_info` interaction succeeds.
- Confirm one `fast_web_search` interaction succeeds.
- Confirm one normal session flow succeeds: `/new`, `/sessions`, `/use <session_id>`.

## Release note

If any item above changes, update the matching smoke instructions and the stable-foundation note in `docs/foundation_baseline.md` in the same patch.
