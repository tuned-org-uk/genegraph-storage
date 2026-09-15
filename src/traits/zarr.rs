//! #4: Zarr-root child trait, the `traits::lance` analog.
//!
//! The kernel-owned `DatasetSummary`/`NodeKind` mirror the Python
//! `DatasetSummary` contract (`base.py`): `dataset_id` is the URL-safe
//! `<label>--<path>` form, `path` is the human-readable rel path (`.` for
//! the root node).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::StorageResult;
use crate::traits::backend::StorageBackend;
use crate::zzarr::ZarrArray;

/// Kind of a scanned Zarr tree node: array or group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeKind {
    /// A readable (v3) or discoverable (v2) array node.
    Array,
    /// A group node; not directly readable.
    Group,
}

impl NodeKind {
    /// String form (`base.py` DatasetSummary.kind values).
    pub const fn as_str(&self) -> &'static str {
        match self {
            NodeKind::Array => "array",
            NodeKind::Group => "group",
        }
    }
}

/// Summary of one Zarr node discovered in a root scan.
///
/// `shape`/`dtype`/`chunks`/`fill_value` describe arrays and are empty for
/// groups; `extra` carries group child counts (`n_arrays`, `n_groups`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetSummary {
    /// URL-safe ID: `<label>--<path segments joined by -->`.
    pub dataset_id: String,
    /// Root label the node belongs to.
    pub root: String,
    /// Human-readable rel path; `.` for the root node itself.
    pub path: String,
    /// Array shape, outermost dimension first; empty for groups.
    pub shape: Vec<u64>,
    /// Data type as recorded in the node metadata (v3 e.g. `float32`,
    /// v2 e.g. `<f4`); empty for groups.
    pub dtype: String,
    /// Chunk shape of the regular grid, when declared.
    pub chunks: Option<Vec<u64>>,
    /// Fill value as recorded (JSON-safe; v3 `NaN`/`Infinity` stay strings).
    pub fill_value: Option<serde_json::Value>,
    /// Node kind.
    pub kind: NodeKind,
    /// Node-kind-specific extras (group child counts).
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// One row replacement for [`ZarrStorageOps::overwrite_vectors`] — the
/// kernel form of the Python `RowUpdate` request model (`usize` makes the
/// `row_index >= 0` constraint structural).
#[derive(Debug, Clone, PartialEq)]
pub struct RowUpdate {
    /// Row to replace; must be below the array's leading extent.
    pub row_index: usize,
    /// Replacement vector; length must equal the array's feature count.
    pub vector: Vec<f64>,
}

/// Zarr-specific operations over a [`ZarrStorage`](crate::zarr_storage::ZarrStorage)
/// root: the child trait analog of [`crate::traits::lance::LanceStorage`].
///
/// Discovery is filesystem-first: the root scan is the source of truth for
/// what a root holds; the kernel registry (`{root}/.arro/metadata.json`)
/// is the catalog view.
pub trait ZarrStorageOps: StorageBackend {
    /// Scans the root recursively and returns one summary per Zarr node
    /// (arrays and groups; v3 `zarr.json` and legacy `.zarray`/`.zgroup`).
    /// A missing root scans to an empty list.
    async fn list_datasets(&self) -> StorageResult<Vec<DatasetSummary>>;

    /// Opens the dataset `dataset_id` as a readable Zarr v3 array handle.
    /// Foreign labels, missing nodes, groups and v2 markers are rejected.
    async fn open(&self, dataset_id: &str) -> StorageResult<ZarrArray>;

    /// Summarizes the single node at `fs_path` (O(1): one metadata read,
    /// no walk). Mirrors the Python `summarize` contract: arrays only.
    async fn summarize(&self, dataset_id: &str, fs_path: &Path) -> StorageResult<DatasetSummary>;

    /// Appends `vecs` (M rows x D features, f64) to the 2-D array
    /// `dataset_id`: resize the leading axis, then write the new rows
    /// (O(M): existing rows are never read). Vectors are auto-cast to the
    /// array's float width (`f64` source, `f32` target narrowing with
    /// `Overflow` above the f32 range); non-float targets are rejected.
    ///
    /// The whole open → validate → resize → write cycle runs under the
    /// dataset's write mailbox: concurrent appends to one dataset
    /// serialize with contiguous, non-overlapping start rows; appends to
    /// different datasets proceed in parallel. Returns
    /// `(start_row, new_nrows)`.
    async fn append_vectors(
        &self,
        dataset_id: &str,
        vecs: &DenseMatrix<f64>,
    ) -> StorageResult<(usize, usize)>;

    /// Replaces specific rows of the 2-D array `dataset_id`
    /// (validate-all-then-write: one invalid row rejects the whole batch
    /// before the first write). Shape is unchanged; duplicate row indices
    /// are allowed and the last entry wins. Returns the number of written
    /// rows (`updates.len()`).
    async fn overwrite_vectors(
        &self,
        dataset_id: &str,
        updates: &[RowUpdate],
    ) -> StorageResult<usize>;
}
