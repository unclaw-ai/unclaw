# Git Repo

Read-only local git repository inspection: branch state, history, and diffs.

Tool hints: Use `git_status` for current state; `git_recent_commits` for history; `git_diff_summary` for changes.

## When to use

Use this skill when the user asks about the state of a local git repository: current branch, uncommitted changes, recent commit history, or what changed between commits or refs.

## How to use

Call `git_status` to report branch, staged/unstaged/untracked file counts, and a compact file list before describing the repository state.

Call `git_recent_commits` to answer questions about recent history or who changed what. Default to the last 10 commits unless the user asks for more.

Call `git_diff_summary` to answer questions about what changed between commits or what is currently modified. Use the `target` argument to compare against a specific commit or ref; default is `HEAD`.

Always set `repo_path` to the relevant repository root. Default `"."` targets the current working directory.

## Safety

Do not infer or report repository state without calling a tool first. Only report file paths, commits, and diff content that are grounded in tool results from this conversation. These tools are strictly read-only — they never modify the repository.
