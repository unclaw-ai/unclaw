# Local Text Search

Search text, code, and documentation files in local directories.

Tool hints: Use `search_local_text` to find occurrences of a term or phrase across local files before reporting findings.

## When to use

Use this skill when the user asks to find a term, phrase, identifier, or pattern across local files: source code, documentation, configuration files, or notes.

## How to use

Call `search_local_text` with a non-empty `query`. Set `root` to the directory to search (defaults to `.`, the current working directory). Narrow the scope with `extensions` when the user asks to search specific file types (e.g. `[".py", ".md"]`).

Answer from the returned matches. Report the file path and line number for each match. Quote the surrounding snippet to ground your answer.

Use `max_results` to limit results when the user expects a small number of matches. Use the default when the user wants a broad search.

## Safety

Do not report file contents or code snippets that are not grounded in a `search_local_text` result from this conversation. This tool is strictly read-only — it never modifies files. It does not access the network.
