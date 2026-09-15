//! #4: Zarr storage backend — a `StorageBackend` sibling of
//! `LanceStorageGraph`, rooted at one Zarr root directory.
//!
//! The root is a forest of Zarr nodes (v3 `zarr.json`, legacy
//! `.zarray`/`.zgroup` markers). Dense/vector writes produce Zarr arrays;
//! operations that do not fit Zarr trees surface
//! [`StorageError::UnsupportedFiletype`] — they are never forced.
//!
//! The kernel-owned metadata registry lives outside user trees at
//! `{root}/.arro/metadata.json` (a `GeneMetadata` JSON, atomically
//! published). Discovery is filesystem-first; the registry is the catalog
//! view.
//!
//! Every dataset write (creation, append, overwrite) runs under the
//! composed write lock of [`crate::commit::try_with_dataset_file_lock`]:
//! the in-process mailbox queues same-process writers, and a fail-fast
//! rendezvous flock at `{root}/.arro/locks/{dataset_id}.lock` excludes
//! foreign processes (POSIX; see the commit module for the off-unix
//! policy).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use log::{info, warn};
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::metadata::FileInfo;
use crate::traits::backend::StorageBackend;
use crate::traits::zarr::{DatasetSummary, NodeKind, RowUpdate, ZarrStorageOps};
use crate::zzarr::ZarrArray;
use crate::{StorageError, StorageResult};

/// Dataset-ID separator of the Python contract (`base.py`).
pub const DATASET_ID_SEP: &str = "--";

/// Kernel-owned registry directory under the root; never scanned, never
/// written into except by the registry itself.
pub const REGISTRY_DIR: &str = ".arro";

/// Registry file name under [`REGISTRY_DIR`].
pub const REGISTRY_FILE: &str = "metadata.json";

/// Kernel registry path for a Zarr root: `{root}/.arro/metadata.json`.
pub fn registry_path(root: &Path) -> PathBuf {
    root.join(REGISTRY_DIR).join(REGISTRY_FILE)
}

/// Encodes `(label, path)` into the URL-safe dataset ID. Slashes and
/// backslashes become `--`; `.` and empty paths collapse to the bare label
/// (Python `make_dataset_id` contract).
pub fn make_dataset_id(label: &str, path: &str) -> String {
    let clean = if path.is_empty() {
        String::new()
    } else {
        path.trim_matches(|c| c == '.' || c == '/')
            .replace('\\', "/")
    };
    let mut id = String::from(label);
    for part in clean.split('/').filter(|p| !p.is_empty()) {
        id.push_str(DATASET_ID_SEP);
        id.push_str(part);
    }
    id
}

/// Decodes a dataset ID back to `(label, rel_path)`; a bare label decodes
/// to rel path `.` (Python `decode_dataset_id` contract).
pub fn decode_dataset_id(id: &str) -> (String, String) {
    let Some((label, rest)) = id.split_once(DATASET_ID_SEP) else {
        return (id.to_string(), ".".to_string());
    };
    let rel = rest.split(DATASET_ID_SEP).collect::<Vec<_>>().join("/");
    (label.to_string(), rel)
}

/// Resolves a root-relative key: rejects empty, absolute and traversing
/// paths instead of joining them against the root.
fn clean_rel(key: &str) -> StorageResult<PathBuf> {
    if key.is_empty() || key.starts_with('/') || key.starts_with('\\') {
        return Err(StorageError::Invalid(format!(
            "invalid relative path '{key}': must be non-empty and root-relative"
        )));
    }
    let mut out = PathBuf::new();
    for seg in key.trim_matches('/').split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                return Err(StorageError::Invalid(format!(
                    "relative path '{key}' must not traverse outside the root"
                )));
            }
            seg if seg.contains(DATASET_ID_SEP) => {
                // Review of #5: a segment carrying `--` would encode into a
                // dataset ID that decodes to a different path; reject it so
                // make/decode_dataset_id stays a bijection.
                return Err(StorageError::Invalid(format!(
                    "relative path segment '{seg}' must not contain '{DATASET_ID_SEP}'"
                )));
            }
            seg => out.push(seg),
        }
    }
    if out.as_os_str().is_empty() {
        return Err(StorageError::Invalid(format!(
            "relative path '{key}' resolves to nothing under the root"
        )));
    }
    Ok(out)
}

/// A storage backend over one Zarr root directory (#4).
///
/// The dataset-ID codec, the registry location and the trait-fit choices
/// are pinned by `src/tests/test_zarr_storage.rs`.
///
/// # Examples
///
/// ```
/// use genegraph_storage::traits::backend::StorageBackend;
/// use genegraph_storage::traits::metadata::Metadata;
/// use genegraph_storage::zarr_storage::ZarrStorage;
/// use smartcore::linalg::basic::arrays::{Array, Array2};
/// use smartcore::linalg::basic::matrix::DenseMatrix;
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let root = std::env::temp_dir().join(format!("genegraph_zarr_doc_{}", std::process::id()));
/// let storage = ZarrStorage::new(root.clone(), "main").unwrap();
///
/// // Seed the kernel registry first: `{root}/.arro/metadata.json`.
/// genegraph_storage::metadata::GeneMetadata::seed_metadata("main", 2, 2, &storage)
///     .await
///     .unwrap();
///
/// let matrix = DenseMatrix::<f64>::from_iterator([1.0, 2.0, 3.0, 4.0].into_iter(), 2, 2, 0);
/// storage.save_dense("cube", &matrix, &storage.metadata_path()).await.unwrap();
/// let loaded = storage.load_dense("cube").await.unwrap();
/// assert_eq!(loaded.shape(), (2, 2));
/// # std::fs::remove_dir_all(&root).ok();
/// # });
/// ```
#[derive(Debug, Clone)]
pub struct ZarrStorage {
    root: PathBuf,
    label: String,
}

impl ZarrStorage {
    /// Creates a backend over `root` labelled `label` (the dataset-ID
    /// prefix). The root may not exist yet; scans of a missing root are
    /// empty and the first save creates the tree.
    pub fn new(root: impl Into<PathBuf>, label: impl Into<String>) -> StorageResult<Self> {
        let label = label.into();
        validate_label(&label)?;
        let root = root.into();
        info!(
            "Creating ZarrStorage at root={}, label={}",
            root.display(),
            label
        );
        Ok(Self { root, label })
    }

    fn registry_path(&self) -> PathBuf {
        registry_path(&self.root)
    }

    /// Traversal-guarded root-relative path for a save/load key.
    fn dataset_path(&self, key: &str) -> StorageResult<PathBuf> {
        Ok(self.root.join(clean_rel(key)?))
    }

    /// Resolves a dataset ID to its directory: label must match this
    /// root, the rel path is traversal-guarded, the node must exist.
    fn dataset_dir(&self, dataset_id: &str) -> StorageResult<PathBuf> {
        let (label, rel) = decode_dataset_id(dataset_id);
        if label != self.label {
            return Err(StorageError::Invalid(format!(
                "dataset '{dataset_id}' does not belong to root '{}'",
                self.label
            )));
        }
        let target = if rel == "." {
            self.root.clone()
        } else {
            self.root.join(clean_rel(&rel)?)
        };
        if !target.is_dir() {
            return Err(StorageError::Invalid(format!(
                "dataset '{dataset_id}' not found at {}",
                target.display()
            )));
        }
        Ok(target)
    }

    /// Registers a written artifact in the kernel registry under its rel
    /// path. Commit-serialized through the registry-path commit actor.
    async fn register_artifact(
        &self,
        rel: &str,
        filetype: &str,
        shape: (usize, usize),
    ) -> StorageResult<()> {
        crate::commit::with_commit_actor(&self.registry_path(), || async {
            let mut md = self.load_metadata().await?;
            let mut info = FileInfo::new(rel.to_string(), filetype, shape, None, None)?;
            // Zarr artifacts are not Lance; record the true storage format.
            info.storage_format = "zzarr".to_string();
            md = md.add_file(rel, info);
            self.save_metadata(&md).await.map(|_| ())
        })
        .await
    }

    /// Blocking load of a 1-D f64 array at an existing rel-path key.
    fn load_f64_1d_blocking(path: PathBuf, key: String) -> StorageResult<Vec<f64>> {
        let arr = crate::zzarr::open(&path)?;
        if arr.shape().len() != 1 {
            return Err(StorageError::Invalid(format!(
                "'{key}' is not a 1-D array (shape {:?})",
                arr.shape()
            )));
        }
        arr.read_all::<f64>()
    }

    async fn load_f64_1d(&self, key: &str) -> StorageResult<Vec<f64>> {
        let path = self.dataset_path(key)?;
        if !path.is_dir() {
            return Err(StorageError::Invalid(format!(
                "vector '{key}' not found at {}",
                path.display()
            )));
        }
        let key = key.to_string();
        let p = path;
        blocking("vector load", move || Self::load_f64_1d_blocking(p, key)).await
    }
}

/// Rejects a label that cannot head a dataset ID.
fn validate_label(label: &str) -> StorageResult<()> {
    if label.is_empty()
        || label.contains(DATASET_ID_SEP)
        || label.contains('/')
        || label.contains('\\')
    {
        return Err(StorageError::Invalid(format!(
            "invalid root label '{label}': must be non-empty and free of \
             '{DATASET_ID_SEP}', '/' and '\\'"
        )));
    }
    Ok(())
}

// =========
// Node metadata reading (v3 + legacy v2 markers)
// =========

#[derive(Debug)]
struct NodeMeta {
    kind: NodeKind,
    shape: Vec<u64>,
    dtype: String,
    chunks: Option<Vec<u64>>,
    fill_value: Option<serde_json::Value>,
}

impl NodeMeta {
    fn group() -> Self {
        Self {
            kind: NodeKind::Group,
            shape: Vec::new(),
            dtype: String::new(),
            chunks: None,
            fill_value: None,
        }
    }
}

fn is_zarr_node(dir: &Path) -> bool {
    dir.join("zarr.json").is_file()
        || dir.join(".zarray").is_file()
        || dir.join(".zgroup").is_file()
}

fn read_json(path: &Path) -> StorageResult<serde_json::Value> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| StorageError::Io(format!("read {}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(StorageError::Serde)
}

fn u64_list(value: &serde_json::Value, dir: &Path, what: &str) -> StorageResult<Vec<u64>> {
    let items = value
        .as_array()
        .ok_or_else(|| unsupported(dir, &format!("node metadata has a malformed '{what}' list")))?;
    items
        .iter()
        .map(|v| {
            v.as_u64().ok_or_else(|| {
                StorageError::UnsupportedFormat(format!(
                    "malformed {what} entry {v} in {}",
                    dir.display()
                ))
            })
        })
        .collect()
}

fn unsupported(dir: &Path, reason: &str) -> StorageError {
    StorageError::UnsupportedFormat(format!("{reason}: {}", dir.display()))
}

fn group_of(v: &serde_json::Value, dir: &Path) -> StorageResult<NodeMeta> {
    match v.get("node_type").and_then(|t| t.as_str()) {
        Some("group") => Ok(NodeMeta::group()),
        Some("array") => {
            let shape = v
                .get("shape")
                .ok_or_else(|| unsupported(dir, "node metadata is missing 'shape'"))?;
            let shape = u64_list(shape, dir, "shape")?;
            let dtype = v
                .get("data_type")
                .and_then(|d| d.as_str())
                .ok_or_else(|| unsupported(dir, "v3 array metadata is missing 'data_type'"))?
                .to_string();
            let chunks = match v
                .get("chunk_grid")
                .and_then(|g| g.get("configuration"))
                .and_then(|c| c.get("chunk_shape"))
            {
                Some(s) => Some(u64_list(s, dir, "chunk_shape")?),
                None => None,
            };
            let fill_value = v.get("fill_value").cloned().filter(|f| !f.is_null());
            Ok(NodeMeta {
                kind: NodeKind::Array,
                shape,
                dtype,
                chunks,
                fill_value,
            })
        }
        _ => Err(unsupported(dir, "v3 zarr.json carries no node_type")),
    }
}

/// Reads the node facts of one directory from its version marker: v3
/// `zarr.json`, v2 `.zarray`/`.zgroup`. Discovery only — no data decode.
fn read_node_meta(dir: &Path) -> StorageResult<NodeMeta> {
    let v3 = dir.join("zarr.json");
    if v3.is_file() {
        return group_of(&read_json(&v3)?, dir);
    }
    let v2 = dir.join(".zarray");
    if v2.is_file() {
        let v = read_json(&v2)?;
        let dtype = v
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| unsupported(dir, "v2 .zarray is missing 'dtype'"))?
            .to_string();
        let chunks = match v.get("chunks") {
            Some(c) => Some(u64_list(c, dir, "chunks")?),
            None => None,
        };
        let fill_value = v.get("fill_value").cloned().filter(|f| !f.is_null());
        let shape = v
            .get("shape")
            .ok_or_else(|| unsupported(dir, "node metadata is missing 'shape'"))?;
        return Ok(NodeMeta {
            kind: NodeKind::Array,
            shape: u64_list(shape, dir, "shape")?,
            dtype,
            chunks,
            fill_value,
        });
    }
    if dir.join(".zgroup").is_file() {
        return Ok(NodeMeta::group());
    }
    Err(unsupported(dir, "not a Zarr node"))
}

/// Direct child directories of `dir`, sorted by name.
fn sorted_children(dir: &Path) -> StorageResult<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| StorageError::Io(format!("read dir {}: {e}", dir.display())))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    Ok(names)
}

fn sub_rel(rel: &str, name: &str) -> String {
    if rel == "." {
        name.to_string()
    } else {
        format!("{rel}/{name}")
    }
}

/// Depth-first collection in the Python `_collect` order: the node itself,
/// then direct arrays (sorted), then each sub-group in turn.
fn collect(label: &str, rel: &str, dir: &Path, out: &mut Vec<DatasetSummary>) -> StorageResult<()> {
    let meta = read_node_meta(dir)?;
    if meta.kind == NodeKind::Array {
        out.push(DatasetSummary {
            dataset_id: make_dataset_id(label, rel),
            root: label.to_string(),
            path: rel.to_string(),
            shape: meta.shape,
            dtype: meta.dtype,
            chunks: meta.chunks,
            fill_value: meta.fill_value,
            kind: NodeKind::Array,
            extra: BTreeMap::new(),
        });
        return Ok(());
    }
    // Group: classify direct node children, then emit with child counts.
    let mut children: Vec<(String, NodeMeta)> = Vec::new();
    for name in sorted_children(dir)? {
        let child = dir.join(&name);
        if is_zarr_node(&child) {
            children.push((name, read_node_meta(&child)?));
        }
    }
    let mut extra = BTreeMap::new();
    extra.insert(
        "n_arrays".to_string(),
        serde_json::json!(
            children
                .iter()
                .filter(|(_, m)| m.kind == NodeKind::Array)
                .count()
        ),
    );
    extra.insert(
        "n_groups".to_string(),
        serde_json::json!(
            children
                .iter()
                .filter(|(_, m)| m.kind == NodeKind::Group)
                .count()
        ),
    );
    out.push(DatasetSummary {
        dataset_id: make_dataset_id(label, rel),
        root: label.to_string(),
        path: rel.to_string(),
        shape: Vec::new(),
        dtype: String::new(),
        chunks: None,
        fill_value: None,
        kind: NodeKind::Group,
        extra,
    });
    for (name, meta) in children.iter().filter(|(_, m)| m.kind == NodeKind::Array) {
        let sub = sub_rel(rel, name);
        out.push(DatasetSummary {
            dataset_id: make_dataset_id(label, &sub),
            root: label.to_string(),
            path: sub,
            shape: meta.shape.clone(),
            dtype: meta.dtype.clone(),
            chunks: meta.chunks.clone(),
            fill_value: meta.fill_value.clone(),
            kind: NodeKind::Array,
            extra: BTreeMap::new(),
        });
    }
    for (name, _) in children.iter().filter(|(_, m)| m.kind == NodeKind::Group) {
        let sub = sub_rel(rel, name);
        collect(label, &sub, &dir.join(name), out)?;
    }
    Ok(())
}

/// Full root scan (sync half of `list_datasets`): the root is either itself
/// a Zarr node (recursed fully) or a container of top-level nodes.
fn scan_root(root: &Path, label: &str) -> StorageResult<Vec<DatasetSummary>> {
    let mut out = Vec::new();
    if !root.exists() {
        warn!("ZarrStorage: root {} does not exist", root.display());
        return Ok(out);
    }
    if is_zarr_node(root) {
        collect(label, ".", root, &mut out)?;
        return Ok(out);
    }
    for name in sorted_children(root)? {
        let child = root.join(&name);
        if is_zarr_node(&child) {
            collect(label, &name, &child, &mut out)?;
        }
    }
    Ok(out)
}

// =========
// Shared write helpers
// =========

/// Row-major flatten of a column-major dense matrix.
fn flatten_row_major(matrix: &DenseMatrix<f64>) -> StorageResult<Vec<f64>> {
    let (rows, cols) = matrix.shape();
    if rows == 0 || cols == 0 {
        return Err(StorageError::Invalid(
            "cannot store an empty dense matrix".to_string(),
        ));
    }
    let mut values = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            values.push(*matrix.get((r, c)));
        }
    }
    Ok(values)
}

/// Blocking load of a 2-D f64 Zarr array into a column-major `DenseMatrix`.
/// Non-Zarr paths surface [`StorageError::UnsupportedFormat`] (never guessed).
fn load_dense_zarr_blocking(path: PathBuf) -> StorageResult<DenseMatrix<f64>> {
    let arr = crate::zzarr::open(&path)?;
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(StorageError::Invalid(format!(
            "expected a 2-D Zarr array at {path:?}, found shape {shape:?}"
        )));
    }
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    let values = arr.read_all::<f64>()?;
    let mut data = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[c * rows + r] = values[r * cols + c];
        }
    }
    DenseMatrix::new(rows, cols, data, true).map_err(|e| StorageError::Invalid(e.to_string()))
}

/// Async wrapper for [`load_dense_zarr_blocking`] (blocking pool).
async fn load_dense_zarr(path: &Path) -> StorageResult<DenseMatrix<f64>> {
    let p = path.to_path_buf();
    blocking("dense load", move || load_dense_zarr_blocking(p)).await
}

/// Checked `usize -> i64` conversion (#51: Overflow, never truncation).
fn checked_i64_values(values: &[usize]) -> StorageResult<Vec<i64>> {
    values
        .iter()
        .map(|&v| {
            i64::try_from(v).map_err(|_| {
                StorageError::Overflow(format!(
                    "index value {v} exceeds i64::MAX and would be silently truncated"
                ))
            })
        })
        .collect()
}

/// f64 -> f32 narrowing for vector writes (the same_kind auto-cast of the
/// Python contract). Finite values above the f32 range surface
/// [`StorageError::Overflow`] instead of becoming silent infinities (#51).
fn cast_f64_to_f32(values: &[f64]) -> StorageResult<Vec<f32>> {
    values
        .iter()
        .map(|&v| {
            let narrowed = v as f32;
            if v.is_finite() && !narrowed.is_finite() {
                return Err(StorageError::Overflow(format!(
                    "value {v} exceeds the f32 range and would be silently narrowed"
                )));
            }
            Ok(narrowed)
        })
        .collect()
}

/// Sync append core: open, validate, resize, write the tail — all under
/// the dataset's composed write lock (in-process mailbox + cross-process
/// rendezvous flock). Python `append_vectors` port (#5): the two-phase
/// validate/re-validate collapses into one phase because validation
/// itself runs under the lock.
fn append_vectors_blocking(
    path: PathBuf,
    lock_file: PathBuf,
    dataset_id: String,
    values: Vec<f64>,
    m: usize,
    d: usize,
) -> StorageResult<(usize, usize)> {
    crate::commit::try_with_dataset_file_lock(&lock_file, &path, || {
        let mut arr = crate::zzarr::open(&path)?;
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(StorageError::Invalid(format!(
                "dataset '{dataset_id}' is not a 2-D array (shape {shape:?})"
            )));
        }
        let (n, arr_d) = (shape[0] as usize, shape[1] as usize);
        if arr_d != d {
            return Err(StorageError::DimensionMismatch {
                expected: format!("{} features", arr_d),
                found: format!("{} features", d),
            });
        }
        match read_node_meta(&path)?.dtype.as_str() {
            "float64" => arr.append(values.as_slice())?,
            "float32" => {
                let cast = cast_f64_to_f32(&values)?;
                arr.append(cast.as_slice())?
            }
            other => {
                return Err(StorageError::Invalid(format!(
                    "dtype mismatch for '{dataset_id}': vectors are f64, \
                     dataset expects '{other}'"
                )));
            }
        }
        Ok((n, n + m))
    })
}

/// Sync overwrite core: validate all updates, then write row by row
/// under the dataset's composed write lock (Python `overwrite_vectors`
/// port; shape unchanged, last duplicate wins).
fn overwrite_vectors_blocking(
    path: PathBuf,
    lock_file: PathBuf,
    dataset_id: String,
    updates: Vec<RowUpdate>,
) -> StorageResult<usize> {
    crate::commit::try_with_dataset_file_lock(&lock_file, &path, || {
        let mut arr = crate::zzarr::open(&path)?;
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(StorageError::Invalid(format!(
                "dataset '{dataset_id}' is not a 2-D array (shape {shape:?})"
            )));
        }
        let (n, d) = (shape[0] as u64, shape[1] as u64);
        // Validate ALL updates before the first write: no partial writes.
        for upd in &updates {
            if upd.row_index as u64 >= n {
                return Err(StorageError::Invalid(format!(
                    "row index {} is out of bounds: dataset '{dataset_id}' has \
                     {n} rows (valid range 0..{n})",
                    upd.row_index
                )));
            }
            if upd.vector.len() as u64 != d {
                return Err(StorageError::DimensionMismatch {
                    expected: format!("{d} features"),
                    found: format!("{} features", upd.vector.len()),
                });
            }
        }
        match read_node_meta(&path)?.dtype.as_str() {
            "float64" => {
                for upd in &updates {
                    write_row(&mut arr, upd.row_index as u64, d, upd.vector.as_slice())?;
                }
            }
            "float32" => {
                let casted: Vec<Vec<f32>> = updates
                    .iter()
                    .map(|upd| cast_f64_to_f32(&upd.vector))
                    .collect::<StorageResult<_>>()?;
                for (upd, vector) in updates.iter().zip(casted) {
                    write_row(&mut arr, upd.row_index as u64, d, vector.as_slice())?;
                }
            }
            other => {
                return Err(StorageError::Invalid(format!(
                    "dtype mismatch for '{dataset_id}': vectors are f64, \
                     dataset expects '{other}'"
                )));
            }
        }
        Ok(updates.len())
    })
}

/// Writes one full row of a 2-D array (any float element type).
fn write_row<T: zarrs::array::Element>(
    arr: &mut ZarrArray,
    row: u64,
    d: u64,
    values: &[T],
) -> StorageResult<()> {
    arr.write_subset(&[row..row + 1, 0..d], values)
}

async fn blocking<T, F>(what: &str, f: F) -> StorageResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> StorageResult<T> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| StorageError::Io(format!("{what} task failed: {e}")))?
}

impl StorageBackend for ZarrStorage {
    fn get_base(&self) -> String {
        self.root.to_string_lossy().into_owned()
    }

    fn get_name(&self) -> String {
        self.label.clone()
    }

    fn base_path(&self) -> PathBuf {
        self.root.clone()
    }

    /// The kernel-owned registry: `{root}/.arro/metadata.json` (#4).
    fn metadata_path(&self) -> PathBuf {
        self.registry_path()
    }

    /// Reports only the kernel registry, never `*_metadata.json` files in
    /// the user tree.
    fn exists(path: &str) -> (bool, Option<PathBuf>) {
        let p = registry_path(Path::new(path));
        if p.is_file() {
            (true, Some(p))
        } else {
            (false, None)
        }
    }

    fn basepath_to_uri(&self) -> StorageResult<String> {
        Self::path_to_uri(&self.root)
    }

    /// Root-joined path for a logical key. Save/load paths validate through
    /// [`Self::dataset_path`]; this accessor mirrors the trait contract for
    /// name resolution without guarding.
    fn file_path(&self, key: &str) -> PathBuf {
        self.root.join(key)
    }

    // =========
    // Dense
    // =========

    /// Saves a dense matrix as a 2-D Zarr f64 array at the rel-path key
    /// (single chunk). Overwrites are rejected.
    async fn save_dense(
        &self,
        key: &str,
        matrix: &DenseMatrix<f64>,
        md_path: &Path,
    ) -> StorageResult<()> {
        self.validate_initialized(md_path)?;
        let path = self.dataset_path(key)?;
        let (rows, cols) = matrix.shape();
        info!("Saving dense {key}: {rows}x{cols} at {}", path.display());
        let values = flatten_row_major(matrix)?;
        let p = path.clone();
        let lock = self.dataset_lock_file_for_key(key);
        blocking("dense save", move || {
            // Creation runs under the dataset's composed write lock
            // (review of #5): the exists-check + write sequence must not
            // interleave, in-process or cross-process.
            crate::commit::try_with_dataset_file_lock(&lock, &p, || {
                crate::zzarr::write_array(
                    &p,
                    &[rows as u64, cols as u64],
                    &[rows as u64, cols as u64],
                    &values,
                    true,
                )
            })
        })
        .await?;
        self.register_artifact(key, "dense", (rows, cols)).await
    }

    async fn load_dense(&self, key: &str) -> StorageResult<DenseMatrix<f64>> {
        let path = self.dataset_path(key)?;
        if !path.is_dir() {
            return Err(StorageError::Invalid(format!(
                "dense '{key}' not found at {}",
                path.display()
            )));
        }
        load_dense_zarr(&path).await
    }

    /// Loads a 2-D f64 Zarr array from an explicit path. Non-Zarr paths
    /// surface [`StorageError::UnsupportedFormat`] (never guessed).
    async fn load_dense_from_file(&self, path: &Path) -> StorageResult<DenseMatrix<f64>> {
        load_dense_zarr(path).await
    }

    /// Writes a dense matrix as a Zarr array at an explicit path
    /// (registry-free). Parent directories are created as needed. No
    /// root-scoped rendezvous exists for raw paths: serialization is the
    /// in-process mailbox only; cross-process callers own their locking.
    async fn save_dense_to_file(data: &DenseMatrix<f64>, path: &Path) -> StorageResult<()> {
        let (rows, cols) = data.shape();
        let values = flatten_row_major(data)?;
        let p = path.to_path_buf();
        blocking("dense save", move || {
            crate::commit::with_dataset_write_lock(&p, || {
                crate::zzarr::write_array(
                    &p,
                    &[rows as u64, cols as u64],
                    &[rows as u64, cols as u64],
                    &values,
                    true,
                )
            })
        })
        .await
    }

    // =========
    // Scalars and indices
    // =========

    /// Saves a 1-D f64 sequence (lambdas, norms, generic vectors).
    async fn save_vector(&self, key: &str, vector: &[f64], md_path: &Path) -> StorageResult<()> {
        self.validate_initialized(md_path)?;
        let len = vector.len();
        let path = self.dataset_path(key)?;
        let values = vector.to_vec();
        let p = path.clone();
        let lock = self.dataset_lock_file_for_key(key);
        blocking("vector save", move || {
            crate::commit::try_with_dataset_file_lock(&lock, &p, || {
                crate::zzarr::write_array(&p, &[len as u64], &[len as u64], &values, true)
            })
        })
        .await?;
        self.register_artifact(key, "vector", (len, 1)).await
    }

    async fn load_vector(&self, key: &str) -> StorageResult<Vec<f64>> {
        self.load_f64_1d(key).await
    }

    async fn save_lambdas(&self, lambdas: &[f64], md_path: &Path) -> StorageResult<()> {
        self.save_vector("lambdas", lambdas, md_path).await
    }

    async fn load_lambdas(&self) -> StorageResult<Vec<f64>> {
        self.load_vector("lambdas").await
    }

    /// Saves indices as an i64 array; values above `i64::MAX` surface
    /// [`StorageError::Overflow`] instead of truncating (#51).
    async fn save_index(&self, key: &str, vector: &[usize], md_path: &Path) -> StorageResult<()> {
        self.validate_initialized(md_path)?;
        let values = checked_i64_values(vector)?;
        let len = values.len();
        let path = self.dataset_path(key)?;
        let p = path.clone();
        let lock = self.dataset_lock_file_for_key(key);
        blocking("index save", move || {
            crate::commit::try_with_dataset_file_lock(&lock, &p, || {
                crate::zzarr::write_array(&p, &[len as u64], &[len as u64], &values, true)
            })
        })
        .await?;
        self.register_artifact(key, "vector", (vector.len(), 1))
            .await
    }

    async fn load_index(&self, key: &str) -> StorageResult<Vec<usize>> {
        let path = self.dataset_path(key)?;
        if !path.is_dir() {
            return Err(StorageError::Invalid(format!(
                "index '{key}' not found at {}",
                path.display()
            )));
        }
        let what = key.to_string();
        let p = path.clone();
        let values = blocking("index load", move || {
            let arr = crate::zzarr::open(&p)?;
            if arr.shape().len() != 1 {
                return Err(StorageError::Invalid(format!(
                    "index '{what}' is not a 1-D array (shape {:?})",
                    arr.shape()
                )));
            }
            arr.read_all::<i64>()
        })
        .await?;
        values
            .iter()
            .map(|&v| {
                usize::try_from(v).map_err(|_| {
                    StorageError::Invalid(format!("negative index value {v} cannot load as usize"))
                })
            })
            .collect()
    }

    // =========
    // Operations that do not fit Zarr trees
    // =========

    async fn save_sparse(
        &self,
        _key: &str,
        _matrix: &sprs::CsMat<f64>,
        _md_path: &Path,
    ) -> StorageResult<()> {
        Err(unsupported_filetype(
            "sparse matrices do not fit Zarr trees; use the Lance backend \
             or zzarr::csr",
        ))
    }

    async fn load_sparse(&self, _key: &str) -> StorageResult<sprs::CsMat<f64>> {
        Err(unsupported_filetype(
            "sparse matrices do not fit Zarr trees; use the Lance backend \
             or zzarr::csr",
        ))
    }

    async fn save_vectors_with(
        &self,
        _name: &str,
        _batch: &arrow::record_batch::RecordBatch,
        _properties: &BTreeMap<String, String>,
        _md_path: &Path,
    ) -> StorageResult<()> {
        Err(unsupported_filetype(
            "RecordBatch vector-space collections do not fit Zarr trees; \
             save 2-D arrays with save_dense",
        ))
    }

    async fn load_vectors(&self, _name: &str) -> StorageResult<arrow::record_batch::RecordBatch> {
        Err(unsupported_filetype(
            "RecordBatch vector-space collections do not fit Zarr trees",
        ))
    }

    async fn save_graph_with(
        &self,
        _name: &str,
        _edges: &[crate::graph::GraphEdge],
        _options: &crate::graph::GraphWriteOptions,
        _md_path: &Path,
    ) -> StorageResult<()> {
        Err(unsupported_filetype(
            "graph collections do not fit Zarr trees; use the Lance backend",
        ))
    }

    async fn load_graph(&self, _name: &str) -> StorageResult<crate::graph::StoredGraph> {
        Err(unsupported_filetype(
            "graph collections do not fit Zarr trees; use the Lance backend",
        ))
    }

    async fn save_vectors_to_path(
        &self,
        _path: &Path,
        _batch: &arrow::record_batch::RecordBatch,
        _properties: &BTreeMap<String, String>,
    ) -> StorageResult<()> {
        Err(unsupported_filetype(
            "RecordBatch vector-space collections do not fit Zarr trees",
        ))
    }

    async fn load_vectors_from_path(
        &self,
        _path: &Path,
    ) -> StorageResult<arrow::record_batch::RecordBatch> {
        Err(unsupported_filetype(
            "RecordBatch vector-space collections do not fit Zarr trees",
        ))
    }

    async fn save_graph_to_path(
        &self,
        _path: &Path,
        _edges: &[crate::graph::GraphEdge],
        _options: &crate::graph::GraphWriteOptions,
    ) -> StorageResult<()> {
        Err(unsupported_filetype(
            "graph collections do not fit Zarr trees; use the Lance backend",
        ))
    }

    async fn load_graph_from_path(&self, _path: &Path) -> StorageResult<crate::graph::StoredGraph> {
        Err(unsupported_filetype(
            "graph collections do not fit Zarr trees; use the Lance backend",
        ))
    }

    async fn load_graph_from_path_with_options(
        &self,
        _path: &Path,
        _options: &crate::graph::GraphReadOptions,
    ) -> StorageResult<crate::graph::StoredGraph> {
        Err(unsupported_filetype(
            "graph collections do not fit Zarr trees; use the Lance backend",
        ))
    }

    async fn load_scalars(&self, _name: &str) -> StorageResult<Vec<f64>> {
        Err(unsupported_filetype(
            "kind-gated scalar collections do not fit Zarr trees; load 1-D \
             arrays with load_vector",
        ))
    }

    async fn save_scalars_to_path(&self, _path: &Path, _values: &[f64]) -> StorageResult<()> {
        Err(unsupported_filetype(
            "kind-stamped scalar collections do not fit Zarr trees; save \
             1-D arrays with save_vector",
        ))
    }

    async fn load_scalars_from_path(&self, _path: &Path) -> StorageResult<Vec<f64>> {
        Err(unsupported_filetype(
            "kind-stamped scalar collections do not fit Zarr trees",
        ))
    }

    async fn collection_schema_from_path(
        &self,
        _path: &Path,
    ) -> StorageResult<arrow::datatypes::Schema> {
        Err(unsupported_filetype(
            "collection schemas are a Lance-format concept; Zarr nodes are \
             summarized with ZarrStorageOps::summarize",
        ))
    }

    /// Collection verification is a lancefmt footer check; Zarr nodes have
    /// no stamped schema footer.
    async fn verify_collection_from_path(
        &self,
        _path: &Path,
        _expected: &crate::lancefmt::verify::DatasetFacts,
    ) -> StorageResult<()> {
        Err(unsupported_filetype(
            "collection verification is a Lance-format concept",
        ))
    }
}

fn unsupported_filetype(what: &str) -> StorageError {
    StorageError::UnsupportedFiletype(what.to_string())
}

impl ZarrStorageOps for ZarrStorage {
    async fn list_datasets(&self) -> StorageResult<Vec<DatasetSummary>> {
        let root = self.root.clone();
        let label = self.label.clone();
        blocking("root scan", move || scan_root(&root, &label)).await
    }

    async fn open(&self, dataset_id: &str) -> StorageResult<ZarrArray> {
        let target = self.dataset_dir(dataset_id)?;
        // Typed group rejection; zzarr::open would only report "not an array".
        if let Ok(meta) = read_node_meta(&target)
            && meta.kind == NodeKind::Group
        {
            return Err(StorageError::Invalid(format!(
                "dataset '{dataset_id}' is a group, not an array"
            )));
        }
        let path = target;
        blocking("array open", move || crate::zzarr::open(&path)).await
    }

    async fn summarize(&self, dataset_id: &str, fs_path: &Path) -> StorageResult<DatasetSummary> {
        let (label, rel) = decode_dataset_id(dataset_id);
        let path = fs_path.to_path_buf();
        let id = dataset_id.to_string();
        blocking("node summarize", move || {
            if !path.is_dir() {
                return Err(StorageError::Invalid(format!(
                    "dataset '{id}' not found at {}",
                    path.display()
                )));
            }
            let meta = read_node_meta(&path)?;
            if meta.kind != NodeKind::Array {
                return Err(StorageError::Invalid(format!(
                    "dataset '{id}' is a group, not an array"
                )));
            }
            Ok(DatasetSummary {
                dataset_id: id,
                root: label,
                path: rel,
                shape: meta.shape,
                dtype: meta.dtype,
                chunks: meta.chunks,
                fill_value: meta.fill_value,
                kind: meta.kind,
                extra: BTreeMap::new(),
            })
        })
        .await
    }

    async fn append_vectors(
        &self,
        dataset_id: &str,
        vecs: &DenseMatrix<f64>,
    ) -> StorageResult<(usize, usize)> {
        let (m, d) = vecs.shape();
        if m == 0 {
            return Err(StorageError::Invalid(
                "Cannot append zero vectors".to_string(),
            ));
        }
        let path = self.dataset_dir(dataset_id)?;
        let values = flatten_row_major(vecs)?;
        let id = dataset_id.to_string();
        let lock = self.dataset_lock_file_for_id(dataset_id);
        let p = path.clone();
        let (start, new_n) = blocking("append vectors", move || {
            append_vectors_blocking(p, lock, id, values, m, d)
        })
        .await?;
        // Best-effort registry refresh (Python `register_dataset` cache
        // update); unregistered user trees stay registry-free. The refresh
        // re-reads the shape from disk inside the commit-actor cycle, so
        // refreshes of concurrent appends converge to the on-disk truth.
        self.update_registered_shape(&self.rel_of(dataset_id))
            .await?;
        Ok((start, new_n))
    }

    async fn overwrite_vectors(
        &self,
        dataset_id: &str,
        updates: &[RowUpdate],
    ) -> StorageResult<usize> {
        if updates.is_empty() {
            return Err(StorageError::Invalid(
                "empty updates: overwrite requires at least one row".to_string(),
            ));
        }
        let path = self.dataset_dir(dataset_id)?;
        let id = dataset_id.to_string();
        let lock = self.dataset_lock_file_for_id(dataset_id);
        let owned = updates.to_vec();
        let p = path;
        blocking("overwrite vectors", move || {
            overwrite_vectors_blocking(p, lock, id, owned)
        })
        .await
    }
}

impl ZarrStorage {
    /// Rel-path key of a dataset ID under this root (post-validation).
    fn rel_of(&self, dataset_id: &str) -> String {
        let (_, rel) = decode_dataset_id(dataset_id);
        rel
    }

    /// Rendezvous lock file of a validated dataset ID, re-derived from
    /// the decoded rel path so a crafted ID can never steer the lock path
    /// outside the kernel namespace: `{root}/.arro/locks/{canonical}.lock`.
    fn dataset_lock_file_for_id(&self, dataset_id: &str) -> PathBuf {
        let rel = self.rel_of(dataset_id);
        let canonical = make_dataset_id(&self.label, &rel);
        self.root
            .join(REGISTRY_DIR)
            .join("locks")
            .join(format!("{canonical}.lock"))
    }

    /// Rendezvous lock file of a save key (same convention, same cycle:
    /// creation and appends to one dataset exclude each other).
    fn dataset_lock_file_for_key(&self, key: &str) -> PathBuf {
        let canonical = make_dataset_id(&self.label, key);
        self.root
            .join(REGISTRY_DIR)
            .join("locks")
            .join(format!("{canonical}.lock"))
    }

    /// Registered shape `(rows, cols)` read from the array's own
    /// `zarr.json` — 1-D arrays register as `(len, 1)`; other ranks are
    /// left untouched (best-effort refresh covers 1-D/2-D artifacts).
    async fn disk_shape(&self, key: &str) -> StorageResult<Option<(usize, usize)>> {
        let path = self.dataset_path(key)?;
        blocking("shape read", move || {
            let meta = read_node_meta(&path)?;
            Ok(match meta.shape.as_slice() {
                [n] => Some((*n as usize, 1)),
                [rows, cols] => Some((*rows as usize, *cols as usize)),
                _ => None,
            })
        })
        .await
    }

    /// Updates the registered shape of `key` in the kernel registry
    /// (Python `register_dataset` cache update). Best-effort in BOTH
    /// directions: unseeded roots and unregistered keys are left alone,
    /// and a save failure is logged, never propagated — the tail write
    /// already succeeded, and surfacing an append failure here would make
    /// a retrying client append twice (#5 review). Unregistered keys are
    /// not republished.
    ///
    /// The shape is read from disk INSIDE the commit-actor cycle: the
    /// refresh runs outside the dataset mailbox, so refreshes of
    /// concurrent appends may land in any order, and reading the disk
    /// truth makes every order converge (the filesystem scan remains the
    /// discovery truth regardless).
    pub(crate) async fn update_registered_shape(&self, key: &str) -> StorageResult<()> {
        if !self.registry_path().is_file() {
            return Ok(());
        }
        let key = key.to_string();
        let outcome = crate::commit::with_commit_actor(&self.registry_path(), || async {
            let mut md = match self.load_metadata().await {
                Ok(md) => md,
                Err(_) => return Ok(false),
            };
            let Some(info) = md.files.get_mut(&key) else {
                return Ok(false);
            };
            let Some((rows, cols)) = self.disk_shape(&key).await? else {
                return Ok(false);
            };
            info.rows = rows;
            info.cols = cols;
            self.save_metadata(&md).await.map(|_| true)
        })
        .await;
        match outcome {
            Ok(_) => {}
            Err(e) => {
                log::warn!(
                    "registry shape refresh for '{key}' failed (best-effort, \
                     disk write stands): {e}"
                );
            }
        }
        Ok(())
    }
}
