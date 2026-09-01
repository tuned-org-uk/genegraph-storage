# Changelog

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
