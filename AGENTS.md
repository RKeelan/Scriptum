# AGENTS.md

## Dependency Management

Always pin dependencies to exact versions — no `^`, `~`, or bare package names. `bunfig.toml` sets `exact = true` under `[install]` so `bun add` pins automatically. `bun.lock` must be committed.
