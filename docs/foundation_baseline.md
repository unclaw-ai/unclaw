# Foundation Baseline

This note marks the current runtime foundation as a regression-sensitive baseline.

## Stable subsystems

Treat these as stable unless a change is explicitly planned and reviewed:

- agent runtime loop in `src/unclaw/core/runtime.py`,
- built-in tool registry and dispatcher flow,
- model-profile resolution and local profile overrides,
- current web stack, including grounded search and `fast_web_search`,
- local storage and workspace behavior under `data/` and configured file roots,
- memory layer: session history, summaries, chat mirrors, and long-term store,
- CLI control surface: `/control`, `/profiles`, `/ctx`, `/model`, `/think`.

## Current operator rules

- `/control` changes local file and terminal access policy immediately in the current CLI.
- `/ctx` persists the override immediately.
- If `/ctx` changes the active profile, Unclaw attempts an immediate model refresh.
- If that refresh succeeds, the next turn in the same CLI uses the new context window.
- If that refresh fails, the saved value is still guaranteed on the next model reload or CLI restart.

## Change policy

For these subsystems, prefer small hardening changes, targeted tests, and explicit operator messaging over architectural churn.
