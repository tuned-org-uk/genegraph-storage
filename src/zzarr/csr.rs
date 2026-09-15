//! zzarr CSR persistence: the graph-Laplacian artifact set written by the
//! index builder (parity with the Python `_persist_csr` layout).
//!
//! Layout under `{slug}/`: `data.zarr` (f32), `indices.zarr` (i64),
//! `indptr.zarr` (i64), and `meta.json` carrying `nitems`, `nfeatures`,
//! `nclusters`, `csr_shape`. `meta.json` publishes atomically via
//! [`crate::generations::write_json_atomic`].

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::generations::write_json_atomic;
use crate::{StorageError, StorageResult};

use super::{ZzarrElement, write_array};

/// The metadata sidecar of a persisted CSR artifact set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CsrMeta {
    pub nitems: usize,
    pub nfeatures: usize,
    pub nclusters: usize,
    pub csr_shape: [usize; 2],
}

/// A decoded CSR artifact set.
#[derive(Debug, Clone, PartialEq)]
pub struct Csr {
    pub data: Vec<f32>,
    pub indices: Vec<i64>,
    pub indptr: Vec<i64>,
    pub meta: CsrMeta,
}

fn write_component<T: ZzarrElement>(path: &Path, values: &[T]) -> StorageResult<()> {
    let len = values.len() as u64;
    write_array(path, &[len], &[len], values, true)
}

/// Persist a CSR artifact set under `dir`.
pub fn write_csr(
    dir: &Path,
    data: &[f32],
    indices: &[i64],
    indptr: &[i64],
    meta: &CsrMeta,
) -> StorageResult<()> {
    std::fs::create_dir_all(dir).map_err(|e| StorageError::Io(e.to_string()))?;
    write_component(&dir.join("data.zarr"), data)?;
    write_component(&dir.join("indices.zarr"), indices)?;
    write_component(&dir.join("indptr.zarr"), indptr)?;
    write_json_atomic(
        &dir.join("meta.json"),
        &serde_json::to_string(meta).map_err(StorageError::Serde)?,
    )
    .map_err(|e| StorageError::Io(format!("meta.json publish failed: {}", e)))
}

/// Read back a CSR artifact set written by [`write_csr`].
pub fn read_csr(dir: &Path) -> StorageResult<Csr> {
    let data = super::open(&dir.join("data.zarr"))?
        .read_all::<f32>()
        .map_err(|e| StorageError::UnsupportedFormat(format!("data.zarr: {}", e)))?;
    let indices = super::open(&dir.join("indices.zarr"))?
        .read_all::<i64>()
        .map_err(|e| StorageError::UnsupportedFormat(format!("indices.zarr: {}", e)))?;
    let indptr = super::open(&dir.join("indptr.zarr"))?
        .read_all::<i64>()
        .map_err(|e| StorageError::UnsupportedFormat(format!("indptr.zarr: {}", e)))?;
    let text = std::fs::read_to_string(dir.join("meta.json"))
        .map_err(|e| StorageError::Io(format!("meta.json: {}", e)))?;
    let meta: CsrMeta = serde_json::from_str(&text).map_err(StorageError::Serde)?;
    Ok(Csr {
        data,
        indices,
        indptr,
        meta,
    })
}
