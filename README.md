# genegraph-storage

A storage layer for graph-based vector databases.

Implements the **Lance format with an in-house writer/reader** (`lancefmt`, pinned to the Lance v2.1 spec) — no dependency on the official `lance` crate. Parquet interop paths and other formats are available via the `StorageBackend` trait.

Provided functionalities:
* `save_metadata`, `load_metadata`: a simple wrapper for all the data in the directory
* `save_*` / `load_*` dense matrices, sparse matrices, vectors, lambdas, indices
* **named collections** (RFC #81): schema-driven vector spaces (`save_vectors` / `load_vectors`, any column layout with one non-null `FixedSizeList` vector column) and first-class graph storage (`save_graph` / `load_graph`, weighted or topology-only, `u32`/`u64` node ids, `f64` weights by default with an explicit `f32` width available)
* vector-space ↔ graph linkage (a vector space references a graph collection through its `graph` property; resolved in one call with `Catalog::describe_vector_space`)
* **transactional generations**: atomic metadata commits (tmp + fsync + rename), `scoped_generation(n)` handles, generation listing/deletion for sweeps, reader pins
* **catalog contract** (`src/catalog.rs`): `TableDescriptor` + `Catalog` trait mirroring the Lance Namespace / Polaris Generic Table API shape, with `LocalRegistry` over the JSON metadata registry
* parquet interop (`save_dense_to_file` / `load_dense_from_file`)

A storage layer for:
* [`javelin-tui`](https://github.com/tuned-org-uk/javelin-tui): a graph-based vector database Text-Interface and
* [`arrowspace`](https://github.com/Mec-iS/arrowspace-rs): the next iteration of vector search

## Usage

```bash
cargo add genegraph_storage
```

Simple example (kept in sync with the compile-checked doc-test on `LanceStorageGraph`):

```rust
use genegraph_storage::lance_storage_graph::LanceStorageGraph;
use genegraph_storage::metadata::GeneMetadata;
use genegraph_storage::traits::backend::StorageBackend;
use genegraph_storage::traits::metadata::Metadata;
use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;

let base = std::env::temp_dir().join(format!("genegraph_doc_{}", std::process::id()));
let storage = LanceStorageGraph::new(
    base.to_string_lossy().to_string(),
    "doc_example".to_string(),
);

// some 2D data
let dense: Vec<Vec<f64>> = vec![vec![0.1, 0.4], vec![0.5, 0.2], vec![0.03, 0.8]];
let (nitems, nfeatures) = (dense.len(), dense[0].len());
let data = DenseMatrix::<f64>::from_iterator(
    dense.iter().flatten().copied(), nitems, nfeatures, 0);

// seed metadata FIRST to initialize the storage directory
let md = GeneMetadata::seed_metadata("doc_example", nitems, nfeatures, &storage)
    .await
    .unwrap();
let md_path = storage.save_metadata(&md).await.unwrap();

// your data is saved in an efficient Lance format
storage
    .save_dense("my_dataset", &data, &md_path)
    .await
    .unwrap();

// Loading back
let loaded = storage.load_dense("my_dataset").await.unwrap();
assert_eq!(loaded.shape(), (nitems, nfeatures));

std::fs::remove_dir_all(&base).ok();
```

Graphs are stored as edge-list collections and convert to CSR at the API boundary:

```rust
use genegraph_storage::graph::{GraphEdge, GraphWriteOptions, NodeIdWidth};

let edges = vec![GraphEdge::weighted(0, 1, 0.5), GraphEdge::weighted(1, 2, -0.25)];
storage.save_graph("my_graph", &edges, &md_path).await.unwrap();

let graph = storage.load_graph("my_graph").await.unwrap();
let csr = graph.to_csr().unwrap(); // sprs CsMat<f64>
```

Weights default to `f64` — the same width and exactness as the `value`
column of the sparse-matrix artifacts — and are persisted faithfully:
the storage layer makes no domain assumptions, so normalization
transforms (`x/(1+x)`, `1 - exp(-x)`, `atan(x)/(π/2)` — never
dataset-wide min-max, which is incompatible with immutable generations)
belong to the producer. Producers that want a guard against a forgotten
transform can declare
`GraphWriteOptions { weight_range: Some((0.0, 1.0)), .. }` — an opt-in
bounds assertion on the already-computed values. Memory-bound consumers
can declare `weight_type: WeightType::F32` to halve the storage bytes;
values that cannot be stored exactly at the declared width are rejected
instead of being silently narrowed (values beyond the f32 range surface
`Overflow`).

Failure semantics of the registry-coupled collection writes: the artifact
is written first and the registry entry is published only afterwards (the
registry is the single commit point), so a live entry never points at a
missing artifact. If the publish itself fails — or the process dies before
it — the artifact remains as unreferenced residue that a later sweep
reclaims; crash recovery for interrupted saves is an orphan sweep, never
a compensating registry rewrite.

Consumers that run their **own** metadata registry (e.g. an
ArrowSpaceMetadata commit pointer at the instance metadata path) use the
registry-free collection I/O — no `GeneMetadata` read or write occurs,
and registry ownership stays with the caller:

```rust
// dataset-level collection metadata is still stamped (kind, layout
// facts, user properties)
storage
    .save_graph_to_path(&storage.file_path("adj"), &edges, &options)
    .await
    .unwrap();
let graph = storage.load_graph("adj").await.unwrap(); // path-resolved
```

Scalar collections (a single `Float64` column — lambdas, norms, ...) are
`kind=vector-space` and load uniformly through `load_scalars`.

Append-style writers get immutable, atomically-committed generations:

```rust
let gen = storage.scoped_generation(1); // artifacts at {logical}__g1_{key}.lance
```

### Concurrent writers and metadata safety

Every `save_*` call runs its metadata registry update through a per-path
commit actor, so concurrent tasks inside one process never lose updates.
Downstream code that performs its **own** metadata read-modify-write cycles
must use the same serialization. Two levels are public:

- **In-process** — `commit::with_commit_actor(metadata_path, cycle)`
  serializes your cycle against the registry paths of the same metadata
  file.
- **Cross-process** — the commit actor is per-process only. Independent
  processes (e.g. separate CLI invocations) take an advisory lock file
  around the whole cycle, resolved through the blessed convention
  `commit::lock_file_for_metadata(metadata_path)` (`{metadata-stem}.lock`
  next to the metadata file). The composed recipe — file lock held across
  the awaited actor cycle — is
  `commit::with_metadata_file_lock(metadata_path, cycle)`:

```rust
use genegraph_storage::commit::with_metadata_file_lock;

let metadata_path = std::path::Path::new("base/ds__g1_metadata.json");
with_metadata_file_lock(metadata_path, || async {
    // load → mutate → publish here: serialized across processes *and*
    // against in-process save_* cycles
    Ok(())
})
.await
.unwrap();
```

For consumers whose whole cycle is synchronous,
`commit::with_file_lock` takes the same lock file around a sync closure —
it serializes across processes but does not compose with the in-process
actor. Every writer of a given metadata file must take the same lock file
before mutating it — arbitration is only as strong as the convention, and
it is advisory: only cooperating writers are excluded.

**Fail-fast contention (#105).** When the contract is to fail on a
concurrent writer rather than wait, use the non-blocking variants
`commit::try_with_file_lock` (sync closure, same lock file) and
`commit::try_with_metadata_file_lock` (composed with the commit actor,
lock resolved via `lock_file_for_metadata`). Acquisition is
`flock(LOCK_EX | LOCK_NB)`: on contention the call returns immediately
with `StorageError::LockWouldBlock { path }` naming the lock file, which
consumers match on to map contention into their own taxonomy (e.g. CLI
exit code 1). The lock file is still created on demand and left in place.

**Operational note.** The flock wait is unbounded and runs through
`spawn_blocking`: a parked waiter cannot be aborted, and many long-lived
waiters can exhaust the runtime's blocking-thread capacity. That is a
sound trade for metadata commits *if* cycles stay short, contention is
normally brief, nothing under the lock does lengthy compute/network I/O
or waits indefinitely, and you understand shutdown behavior with a stuck
holder (blocked `spawn_blocking` tasks are abandoned by
`Runtime::shutdown_timeout`, awaited by `shutdown_background`; process
exit always releases the flock). If waits could be prolonged or numerous,
prefer a dedicated lock-management thread, an explicit timeout or
cancellation strategy, or a storage system with transactional
coordination.

## Lance format

The default build runs the in-house Lance v2.1 implementation (`lancefmt`) for all `StorageBackend` I/O: manifest with inline Overwrite transactions, txn files, version hints, MiniBlock pages with Flat / InlineBitpacking / FixedSizeList value compression. Encodings outside the supported subset are rejected with `StorageError::UnsupportedFormat` (never guessed). Interop conformance is fixture-based: golden fixtures written by the official lance crate are read back by the in-house reader's suite.

## Extending and traits

Every custom definition of a Lance database (store or manifold or data-cube) should implement the `Metadata` trait (or reuse `GeneMetadata`) and the `StorageBackend` trait like `LanceStorageGraph` does with `GeneMetadata` and `LanceStorage`. Collections additionally surface through the `Catalog` trait (`LocalRegistry` over `GeneMetadata`).

Traits in `traits` module can also be reused to implement other formats. Other formats can use `StorageBackend` to implement similar child-traits alike to `LanceStorage`. Then if matched with a custom `Metadata` instance can make a database, so every database is simply a `StorageBackend + Metadata`.

## Contributing
See `.github/` directory.
