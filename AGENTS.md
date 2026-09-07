# AGENTS.md

## Architecture Overview

`StorageBackend + Metadata` is the core abstraction: every database is a backend implementation paired with a metadata type. All I/O is async (Tokio); library code never blocks or creates its own runtime.

Module map:
- `traits/backend.rs` — `StorageBackend`: async save/load for dense/sparse matrices, scalars, indices, vector-space and graph collections. Most conversions (matrix <-> RecordBatch) live here as provided methods.
- `traits/lance.rs` — `LanceStorage`: Lance-specific child-trait plus shared write-path helpers (reserved-property rejection, collection metadata stamping).
- `traits/metadata.rs` + `metadata.rs` — `Metadata` trait and `GeneMetadata`/`FileInfo`/`CollectionKind`: the JSON registry of every artifact in the storage directory.
- `lance_storage_graph.rs` — `LanceStorageGraph`: the concrete backend (implements both traits). `scoped_generation(n)` returns a handle over generation `N`.
- `lancefmt/` — in-house Lance v2.1 writer/reader (default for all I/O; no official `lance` dependency). `pb/` holds vendored protobuf types; no `protoc` needed. Unsupported encodings are rejected with `StorageError::UnsupportedFormat`, never guessed. Conformance is fixture-based (`tests/fixtures`, golden files from official lance).
- `graph.rs` — graph collection types (`GraphEdge`, `StoredGraph`, `GraphWriteOptions`, `NodeIdWidth`); CSR conversion stays at the API boundary, sprs never reaches the format layer.
- `generations.rs` — transactional generations: immutable `{logical}__g{N}` artifacts, the per-generation metadata JSON as single commit pointer, `write_json_atomic` (tmp + fsync + rename), sweep/delete helpers, reader pins. `__g{digits}` suffix on instance names is reserved.
- `commit.rs` — commit serialization: per-metadata-path and per-dataset-dir mailboxes (duva-style single commit actor) so concurrent writers can't lose updates or mint duplicate manifest versions. Registries are weak-valued to stay bounded.
- `catalog.rs` — `TableDescriptor` + `Catalog` (Lance Namespace / Polaris Generic Table API shape) with `LocalRegistry` over `GeneMetadata`.
- `src/tests/` — unit/integration suites (`cargo test --release --lib`; doc-tests via `cargo test --release --doc`).

Key invariants to preserve when editing:
- Metadata must be seeded before any `save_*` call; `save_metadata` publishes atomically via `write_json_atomic`.
- Reserved properties/metadata keys (`catalog::RESERVED_PROPERTIES`, `graph::RESERVED_METADATA_KEYS`) are computed facts; user properties may not shadow them.
- Values above a storage type's range surface `StorageError::Overflow`, never silent truncation (#51).
- `StorageError` is `#[non_exhaustive]`; downstream matches carry a wildcard arm.

## Shell tools

- Do not use perl: use `python3` (`uv` if present), `sed`, `awk` and
  plain bash commands instead.

## Build & verify

- `cargo test --release` — full suite; always run tests with the
  `--release` flag (debug builds of the lance stack are slow); ledgered
  pre-existing failures live in #23, do not silently skip; run full
  suite only at the end; most of the time run a subset of test cases.
- `cargo clippy --all-targets -- -D warnings` — must be clean before every commit.
- `cargo fmt` — run before committing.

## Testing rules

- TDD strict: failing test first, including reproduction tests for every bug fix.
- Random seed **3407** for NEW tests; legacy fixtures keep 42/9999/123.
- Test dirs: `crate::tests::tmp_dir()` returns `tempfile::TempDir` — hold the guard
  for the whole test; never let it drop while a DB handle points inside.
- Tests are DAMP over DRY: each test reads as a standalone specification.

## Writing style

- Use ASD-STE100 Simplified Technical English in all prose: short sentences, one idea each, active voice, imperative for instructions, approved terminology, no idioms.
- Keep code comments concise: state the contract or the reason, never the narrative. Docs/CHANGELOG entries follow the same rule.
- Make issue bodies detailed: context, design, evidence, acceptance criteria — an issue is the durable record; a comment is not.
