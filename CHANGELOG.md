# Changelog

## 0.61.0 (2026-09-03)

Completes the registry-free collections API (#107) and makes the
metadata-only read path first-class (#109), with an opt-in strict graph
read mode (#108). A registry-owning consumer (genefold-vd Workstream B)
can now save and reload scalar collections with no GeneMetadata I/O and
no hand-written schema metadata, gate on dataset kind/stamps without a
full columnar scan, and detect the unstamped-`num_nodes` hazard at the
API instead of silently resizing.

**Added**

- `save_scalars_to_path` / `load_scalars_from_path` (#107): registry-free
  scalar collection I/O mirroring the reader's exact contract. A single
  non-nullable `Float64` column (neutral default name `lambda`), dataset
  stamped `kind = vector-space`, parent dirs created, no `GeneMetadata`
  read or write. `load_scalars` now delegates to the path-based reader.
- `lancefmt::read_schema` (#109): metadata-only dataset read that parses
  the flat schema (fields + KV metadata) from the newest manifest without
  decoding any column buffers; matches `scan_all(...).schema()`. Surfaced
  on the backend as `collection_schema_from_path` / `collection_schema`.
- Strict graph read mode (#108): `GraphReadOptions` with a `strict` flag,
  plus `load_graph_from_path_with_options` and the
  `load_graph_from_path_strict` convenience variant. Strict mode rejects
  datasets without a `num_nodes` stamp with a typed `StorageError::Invalid`
  (instead of silently resizing to `max_id + 1`) and validates the
  `src`/`dst`/`weight` column names, positively identifying pre-collections
  triplet artifacts (`row`/`col`/`value`) rather than positionally coercing
  them. The tolerant default is unchanged (the compat shim).

**Changed**

- `load_scalars` is now a thin delegate over `load_scalars_from_path`;
  the kind gate and single-column/`Float64|Float32` validation are shared.
- **Breaking**: `StorageBackend` gains the required registry-free methods
  `save_scalars_to_path`, `load_scalars_from_path`,
  `load_graph_from_path_with_options`, `load_graph_from_path_strict` and
  `collection_schema_from_path` (four of which have no default
  implementation), so external implementors of the trait must add them.
  `collection_schema` is provided with a default implementation.

Refs: #107, #108, #109, Genefold/genefold-vd#46.

## 0.60.0 (2026-09-03)

First catalog-complete release line (#106): the catalog contract
(#75 M-C1) becomes end-to-end adoptable — catalog-owning consumers
(genefold-vd Workstream B) can save and load collections with no
GeneMetadata I/O through the registry-free collection writers, graph
weights are schema-declared (`f64` default) with an opt-in compliance
range, and scalar collections are uniformly loadable. Version jumps
from 0.53.x to mark the milestone.

**Fixed**

- Restored artifact-first ordering in `save_vectors_with` /
  `save_graph_with`: the registry publish (through the commit actor) is
  the single commit point, so a live registry entry only exists once the
  artifact is durable at its final location. The 0.53.0 ordering
  (registry before artifact) could commit entries pointing at artifacts
  whose write then failed — worse than the unreferenced residue left by a
  failed publish, which discovery never treats as live and which a later
  sweep reclaims. Regression-guarded by a test that forces the artifact
  write to fail and asserts the registry file is untouched with no live
  entry; the residue case (failed publish) is pinned as the documented
  crash-recovery contract (orphan sweep, never a compensating rewrite).
- `load_scalars` enforces the logical collection kind (#106 review): the
  dataset must be stamped `kind = vector-space`; missing or mismatched
  kinds are rejected (physical shape determines decodability, the
  logical kind determines semantic validity). Legacy scalar writers
  (`save_primitive_column`: lambdas, norms, vectors, indices, centroid
  maps, cluster assignments) now stamp the dataset-level kind, matching
  the registry-level `for_filetype("vector")` shim.

**Changed**

- Graph weight ranges are an opt-in compliance check, not a
  storage-layer assumption (#106): `GraphWriteOptions::weight_range:
  Option<(f64, f64)>` asserts that every weight already lies in the
  declared closed interval (e.g. `Some((0.0, 1.0))` for producers that
  normalize upstream) and rejects violations — NaN lies in no interval —
  with `Invalid`; inverted intervals are rejected configurations. With
  no range declared the crate persists weights faithfully:
  data-independent normalization transforms (`x/(1+x)`, `1-exp(-x)`,
  `atan(x)/(π/2)`) belong to the producer (tracked in
  Genefold/genefold-vd#51), and
  dataset-wide min-max normalization is incompatible with immutable
  generations anyway. The f32 width keeps its #51 narrowing guard:
  finite values beyond the f32 range surface `Overflow`, inexact
  narrowings `Invalid`, infinities narrow bit-exactly and NaN is
  preserved as NaN (classification semantics; full bit-exactness
  requires the f64 width).

**Docs**

- Corrected the `graph.rs` module-level width defaults (`f64` weights,
  not `f32`); README documents the faithful-persistence contract, the
  opt-in `weight_range` check and the failure semantics (single commit
  point, orphan sweep for residue).

## 0.53.0 (2026-09-03)

Collections adoption blockers for genefold-vd Workstream B (#106,
follow-up to the RFC #81 collection API).

**Added**

- Schema-declared graph weight widths (#106): `GraphWriteOptions::weight_type`
  (`WeightType::{F64, F32}`) mirrors `NodeIdWidth`. The default `f64`
  width matches the legacy sparse-matrix `value` column — an f64 CSR
  (laplacian, adjacency) round-trips bit-identically through
  `save_graph`, exactly like through `save_sparse`. The explicit `f32`
  width halves the storage bytes for memory-bound consumers but rejects
  weights that cannot be stored exactly (`Invalid` with guidance towards
  `WeightType::F64`; out-of-range values surface `Overflow`) instead of
  silently narrowing them — the #51 invariant, float edition. `GraphEdge`
  weights are `f64` at the API boundary (losslessly upcast from `f32`
  storage on load), the width is stamped into the dataset schema metadata
  and the registry properties (`weight_type`), and `StoredGraph` reports
  it. This resolves the RFC #81 open question ("Weights: Float64 only, or
  allow Float32?") with "both, schema-declared, f64 default".
- Registry-free collection writes and path-based readers (#106):
  `save_vectors_to_path` / `save_graph_to_path` validate, stamp
  dataset-level collection metadata (kind, layout facts, user properties)
  and write the dataset at an explicit path with **no `GeneMetadata`
  read or write** — registry ownership stays with consumers that run
  their own metadata file (e.g. an ArrowSpaceMetadata commit pointer at
  the instance metadata path). Counterparts
  `load_vectors_from_path` / `load_graph_from_path` complement the
  already registry-free name-based readers. Guarded from the external
  consumer view in `tests/api_public.rs`.
- Scalar collection reader `load_scalars` (#106): a single
  `Float64|Float32` column (lambdas, norms, ...) loads as `Vec<f64>`
  (lossless upcast), so every `kind=vector-space` collection is uniformly
  loadable instead of requiring the legacy fixed-key readers.

**Changed**

- `save_vectors_with` / `save_graph_with` commit the registry entry
  **before** writing the artifact (the `save_sparse` ordering): a failed
  registry step no longer leaves a stray orphan artifact behind (#106,
  spike-verified on a consumer directory whose metadata path holds a
  foreign document).
- **Breaking**: `GraphEdge::weighted` takes `f64` (was `f32`;
  `f32`-origin values pass as `f64::from(w)` and keep round-tripping
  bit-exactly through the default `f64` store); `GraphWriteOptions` and
  `StoredGraph` gain the `weight_type` field; `StorageBackend` gains the
  required registry-free methods above.

## 0.52.0 (2026-09-02)

**Added**

- Non-blocking file-lock variants for fail-fast consumer RMW cycles
  (#105, follow-up to #100): `commit::try_with_file_lock(lock_path, f)`
  and `commit::try_with_metadata_file_lock(metadata_path, cycle)` take
  the same lock files with the same hold scopes as their blocking
  counterparts, but acquisition is non-blocking (`flock(LOCK_EX |
  LOCK_NB)`): on contention the call returns immediately with the new
  distinctly matchable `StorageError::LockWouldBlock { path }` naming
  the lock file, instead of parking the waiter on the blocking pool.
  Advisory (cooperating-writers-only) and rendezvous-point semantics
  (lock file created on demand, left in place) are unchanged and
  documented next to the blocking forms; off unix the try forms fail
  with `StorageError::UnsupportedFormat`.

## 0.51.0 (2026-09-02)

Consumer-review follow-ups from the genefold-vd adoption line
(genegraph-storage #100, #101, #102).

**Added**

- `commit::with_commit_actor` is public: downstream consumers running
  their own metadata read-modify-write cycles serialize against the same
  per-path in-process commit actor the `save_*` registry paths use
  (#100).
- Cross-process arbitration convention (#100): `commit::with_file_lock`
  (advisory `flock`, hold scope = the closure, runs on the blocking
  pool; typed `UnsupportedFormat` off unix) and
  `commit::lock_file_for_metadata` (`{metadata-stem}.lock` next to the
  metadata file). Contract documented in the `commit` module docs and
  the README; `tests/api_public.rs` guards the public surface by
  compiling the crate as an external consumer.

**Changed**

- `generations::write_json_atomic` fsyncs the parent directory after the
  rename, and on a freshly created parent the new directory (and its
  parent) as well, so the metadata commit pointer survives a post-rename
  crash (#101). The directory-fsync helper moves from the lancefmt
  writer to `generations::fsync_dir` as the shared discipline; bare
  filenames (`parent() == ""`) resolve to `.` and keep working.
- `StorageBackend::save_sparse` rejects `nnz=0` matrices early and typed
  (`StorageError::Invalid` — "fully disconnected: adjacency nnz=0")
  before the metadata commit cycle and any artifact write, so a
  disconnected graph leaves no partial directory behind (#102). The
  lancefmt writer's empty-batch guard remains as the format-layer
  backstop.

## 0.26.0 (2026-09-02)

Transactional generations: the durability layer for append-style writers
(genegraph-storage #93, RFC #81 phase P5). Lessons taken from
Migorithm/duva: immutable segments + a single atomic commit pointer.

**Added**

- `generations` module: `{logical}__g{N}` generation naming/parse helpers,
  `list_generations` (committed only — a generation without a metadata
  file was never committed), `list_artifact_generations` (committed +
  orphaned, for sweeps), `delete_generation` (prefix-exact removal of a
  generation's artifacts and metadata), `write_json_atomic` (tmp + fsync +
  rename — the only sanctioned way to publish a metadata file).
- `LanceStorageGraph::scoped_generation(n)`: a handle routing artifact IO
  at `{logical}__g{n}_{key}.lance` with the per-generation metadata file
  as commit pointer; logical identity recoverable via
  `generations::logical_name`.

**Changed**

- `StorageBackend::save_metadata` (default impl) now publishes atomically
  via `write_json_atomic` instead of an in-place write that could be
  observed half-written after a crash.

## 0.25.0 (2026-09-02)

Version line for the post-M5 development cycle (in-house Lance v2.1
implementation shipped and verified in 0.20.0; see #75 close-out and
RFC #81 for the generic vector-space + graph storage plan).

**Removed**

- The `official-lance` feature (M5 transition escape hatch) after one
  release cycle, and the `lance` dev-dependency itself: the crate no
  longer references the official lance crate anywhere in its dependency
  graph (#75 M5 plan). Interop conformance is now fixture-based — the
  golden fixtures written by the official crate (0.13.2) are read back
  by the in-house reader's suite; the fixture generator and
  official-reader round-trip tests were pruned with the dev-dependency.

**Added**

- Ergonomic metadata construction: `GeneMetadata::{new, with_base,
  with_dimensions, add_file, new_fileinfo}` are now inherent methods,
  so the common construction chain no longer requires importing the
  `Metadata` trait (the trait remains for generic code).
- (cycle opens) RFC #81: named collections, generic dense vector
  widths, first-class graph storage (edges + weights), vector-space to
  graph linkage.

## 0.20.0 (2026-09-02)

The in-house Lance milestone (#75 M4 + M5 + M-C1): the default build no
longer depends on the official `lance` crate.

**Breaking**

- `lance` moved out of the default dependency tree (now an optional
  dependency behind the new `official-lance` feature, plus a
  dev-dependency for conformance tests). The default build runs the
  in-house Lance v2.1 implementation (`lancefmt`) for all
  `StorageBackend` I/O. Runtime dependency graph: **542 -> 208 packages;
  datafusion family 30 -> 0**; the only remaining lance-family crate is
  the `lance-bitpacking` leaf.
- New runtime dependencies: `prost`, `prost-types`, `lance-bitpacking`
  (all Apache-2.0).

**Added**

- M4: `lancefmt` overwrite semantics — writing to an existing dataset
  creates a new manifest version (`N+1.manifest`, updated version hint)
  whose fragment set replaces the previous one; readers (ours and
  official) always land on the latest version.
- `Int64` column support in `lancefmt` (schema, writer, reader incl.
  InlineBitpacking decode) — needed by `save_cluster_assignments`.
- M-C1: catalog contract (`src/catalog.rs`): `TableDescriptor` +
  `Catalog` trait mirroring the Lance Namespace / Polaris Generic Table
  API shape (`name`/`format`/`base-location`/`properties`), with
  `LocalRegistry` implemented over the existing JSON metadata registry
  (standards-hygiene per decision D1 in #75; no catalog server client).

**Changed**

- `LanceStorage::write_lance_batch_async` /
  `read_lance_all_batches_async` run `lancefmt` on the blocking pool by
  default; the `official-lance` feature restores the previous behavior
  during the transition cycle.

## 0.13.3 (2026-09-02)

## 0.13.3 (2026-09-02)

**Added**

- M1: in-house Lance v2.1 **writer** (`lancefmt::write_dataset`) for the
  artifact schema subset (`Float64`, `UInt32`, `FixedSizeList<Float64>`,
  non-null): MiniBlock pages with Flat value compression, manifest v1 with
  inline Overwrite transaction, txn file, version hint. Byte-layout mirrors
  the official writer.
- M2: in-house Lance v2.1 **full-scan reader** (`lancefmt::scan_all`):
  manifest open, footer parse, MiniBlock chunk walk; decodes Flat,
  InlineBitpacking (FastLanes via `lance-bitpacking`) and FixedSizeList
  value compressions over all-valid repdef layers.
- 3-way interop proof (15 tests): ours->ours round-trips, official lance 11
  reads our files, our reader reads the official golden fixtures.
- Arrow schema <-> `lance.file.Field` mapping (logical types `double`,
  `uint32`, `fixed_size_list:double:N`) incl. schema-metadata round-trip.
- Generated protobuf types committed under `src/lancefmt/pb/` (prost 0.14,
  provenance headers; consumers need no `protoc`); vendored
  `transaction.proto`.

**Changed**

- New runtime dependencies: `prost`, `prost-types`, `lance-bitpacking`
  (all Apache-2.0; `lance-bitpacking` is a leaf crate).

**Notes**

- `StorageBackend` still routes to the official `lance` crate; the swap to
  `lancefmt` internals is M5 of #75.
- Encodings outside the supported subset are rejected with
  `StorageError::UnsupportedFormat` (never guessed, per #75).

## 0.13.2 (2026-09-02)

**Added**

- M0 of the in-house Lance format plan (#75, re-scoped; framing in #74):
  `lancefmt` module pinning the **Lance v2.1 spec** with vendored protobuf
  definitions (Apache-2.0, from lance 11.0.0, provenance in
  `src/lancefmt/protos/README.md`) and a root `NOTICE`.
- Conformance harness: golden fixtures generated by official lance 11 for our
  artifact schemas (`float64_nonnull`, `float64_multipage`, `uint32_nonnull`,
  `fsl_f64_nonnull`, `sparse_triplet_meta`) plus round-trip tests; fixture
  generator is an explicit `--ignored` test.
- Conformance finding recorded: the official writer does not preserve
  FixedSizeList child-field nullability; documented and tolerated in tests.
- Dependency baseline recorded on #75 (512 crates on main; 30 datafusion
  family, 19 lance family) — re-measured at M5.

**Fixed**

- Windows CI: `test_path_to_uri_absolute_missing_path_is_ok` is now
  platform-aware (drive-letter absolute paths, `file:///C:/...` URIs).

No runtime behavior changes in this release: the `lancefmt` module is
spec + data + tests only. Writer/reader land in M1+M2 (#78), the
`StorageBackend` swap in M5.

## 0.13.1 (2026-09-01)

Issues #45-#55: no more panics in library code; typed errors for all
recoverable conditions.

**Breaking**

- `StorageError` is now `#[non_exhaustive]` and gains variants
  `InvalidState`, `UnsupportedFormat`, `UnsupportedFiletype`,
  `DimensionMismatch { expected, found }`, `Overflow` (#53). Downstream
  `match` expressions need a wildcard arm.
- `FileInfo::which_format` / `FileInfo::which_filetype` return
  `StorageResult<String>` instead of panicking on unknown input (#45).
- `FileInfo::new` returns `StorageResult<Self>`;
  `Metadata::new_fileinfo` returns `StorageResult<FileInfo>` (#45).
- `LanceStorageGraph` fields `_base`/`_name` renamed to `base`/`name`
  (pub(crate) only, no public API impact) (#55).

**Fixed**

- `validate_initialized` returns `StorageError::InvalidState` on metadata
  path mismatch instead of `assert_eq!`-panicking (#47).
- `from_sparse_record_batch` returns `StorageError::DimensionMismatch`
  instead of panicking on schema/storage metadata disagreement (#46).
- `path_to_uri` propagates path resolution failures (permissions, symlink
  loops) as `StorageError::Io`; only `NotFound` still falls back to the
  given absolute path (fresh saves) (#48).
- `save_index` / `save_centroid_map` reject `usize` values above `u32::MAX`
  with `StorageError::Overflow` instead of silently truncating (#51).
- `save_metadata` / `load_metadata` / `GeneMetadata::read` use `tokio::fs`
  instead of blocking `std::fs` inside async fns (#49).
- `load_dense_from_file` parquet path runs on `tokio::task::spawn_blocking`
  instead of blocking the executor (#52).

**Changed**

- Duplicated save pattern across `save_lambdas`, `save_vector`,
  `save_index`, `save_centroid_map`, `save_subcentroid_lambdas`,
  `save_item_norms`, `save_cluster_assignments` consolidated into
  `LanceStorage::save_primitive_column` (~150 lines removed) (#50).
- README quickstart fixed (wrong imports, invalid tuple syntax, wrong
  method order); equivalent example added as a run-on-every-`cargo test`
  doc-test on `LanceStorageGraph` (#54).

## 0.13.0 (2026-09-01)

**Breaking**

- `lance` upgraded 6.0.0 -> 11.0.0 (arrow/parquet stay ^58). Files written by
  0.12.x are readable; re-verify downstream builds against lance 11.
- `StorageBackend::path_to_uri` now returns `StorageResult<String>`:
  relative paths are rejected with a typed error instead of being joined
  against cwd / CARGO_MANIFEST_DIR; existing paths canonicalise; absolute
  non-existing paths are allowed (fresh saves). `basepath_to_uri` fallible.
- `LanceStorage::read_lance_first_batch_async` removed from the trait.
- `smartcore` pinned to 0.6.14 to align with arrowspace 0.26 / genefold-vd
  (single smartcore tree across the boundary; DenseMatrix/CsMat cross it).

**Fixed**

- G2: all 8 artifact load paths read ALL batches (previously truncated at
  the 8192-row scan batch boundary). Repro tests: 10k lambdas, 10k-nnz sparse.
- G7: typed `path_to_uri` via `url::Url::from_file_path` (no lossy fallbacks).

**Consumers**

- genefold-vd adoption pending (Genefold/genefold-vd#23 Workstream B);
  latentfold-core must adapt to the fallible `path_to_uri`.
