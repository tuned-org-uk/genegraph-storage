# Changelog

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
