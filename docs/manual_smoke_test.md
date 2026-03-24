# Manual Smoke Test

This is the lightweight operator smoke pass for the current CLI/runtime foundation.

## 1. Startup and preflight

Run `unclaw start`.

Confirm the banner and preflight report show:

- the active model pack,
- the default profile,
- the current control preset,
- the profile/context summary,
- the required model status.

Exit with `Ctrl-D` if you only wanted a startup check.

## 2. Control surface

Inside the CLI, run:

- `/control safe`
- `/control workspace`
- `/control full`

For each command, confirm the reply:

- says the preset was saved,
- says the new tool-access rule applies immediately in this CLI,
- shows the allowed roots summary.

Return to the normal preset you want before continuing.

## 3. Context override rule

Inside the CLI, run `/ctx` first and note the active profile.

Then run `/ctx <active_profile> <num_ctx>`.

Expected rule:

- the override is saved immediately,
- if the edited profile is active, Unclaw immediately attempts a model refresh,
- on refresh success, the CLI says the next turn will use the new context window,
- on refresh failure, the CLI says the new value is guaranteed on next model reload or CLI restart.

Also run `/ctx <inactive_profile> <num_ctx>` once and confirm the CLI says the value will apply the next time that profile is loaded.

## 4. Profiles output

Run `/profiles`.

Confirm the active profile is marked and each entry shows:

- profile name,
- model name,
- `ctx=...`,
- tool mode.

## 5. `system_info`

Ask a local-machine question such as:

`What machine and Python runtime are you running on right now? Use system_info.`

Confirm the reply is grounded in current local facts, not memory.

## 6. `fast_web_search`

Ask a literal entity lookup such as:

`Who is Marine Leleu? Use fast_web_search if needed.`

Confirm the reply is grounded and that the logs show a `fast_web_search` tool call if the model chose the tool path.

## 7. Session flow

Run:

- `/new`
- `/sessions`
- `/use <session_id>`

Confirm session creation, listing, and switching all still work cleanly.
