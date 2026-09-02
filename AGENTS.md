# AGENTS.md

## Shell tools

- Do not use perl for file editing/transformations. Use python3 (via `uv` if present), sed, awk and bash commands instead.

## Build & test

- Run cargo test always with the `--release` flag (e.g. `cargo test --release --lib`).
