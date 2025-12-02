use log::debug;
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::CsMat;
use std::path::{Path, PathBuf};

use crate::StorageResult;
use crate::metadata::GeneMetadata;

/// Storage backend trait for persisting ArrowSpace graph embeddings.
///
/// ## Initialization Protocol
///
/// Storage must be initialized before data can be saved:
///
/// 1. Call `save_metadata()` or `save_eigenmaps_all()`/`save_energymaps_all()` first
/// 2. Only then can individual `save_dense()`, `save_sparse()`, `save_lambdas()` be called
/// 3. All save operations validate that metadata exists
///
/// Filename format is : <base dir> / <instance name>_<stem>.lance
///
/// ## Async usage
///
/// This trait is now async-first for all I/O methods. Implementations (e.g. `LanceStorage`)
/// must integrate with Tokio by providing non-blocking async methods; no `block_on` or
/// nested runtimes are used inside the backend.
///
/// ## Example
///
/// ```ignore
/// let storage = LanceStorage::new(base, name);
/// let builder = ArrowSpaceBuilder::new();
/// let (mut aspace, gl) = builder.build_for_persistence(data, "Eigen", None);
///
/// // This initializes the storage directory with metadata and writes all artifacts
/// storage.save_eigenmaps_all(&builder, &mut aspace, &gl).await?;
///
/// // Now individual loads will work
/// let raw = storage.load_dense("rawinput").await?;
/// ```
pub trait StorageBackend: Send + Sync {
    /// Base directory of the instance
    fn get_base(&self) -> String;
    /// Name of the instance
    fn get_name(&self) -> String;

    fn path_to_uri(path: &Path) -> String;

    ///
    /// Returns `true` and the path to the metadata file if metadata file exists and is valid,
    /// `false` otherwise.
    /// This is used to avoid overwriting existing indexes.
    fn exists(path: &str) -> (bool, Option<PathBuf>) {
        let base_path = std::path::PathBuf::from(path);
        if !base_path.exists() {
            debug!("StorageBackend: path {:?} does not exist", base_path);
            return (false, None);
        }

        // Check for any _metadata.json file in the directory
        if let Ok(entries) = std::fs::read_dir(&base_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.ends_with("_metadata.json") {
                        debug!("StorageBackend::exists: found metadata file at {:?}", path);
                        return (true, Some(path));
                    }
                }
            }
        }
        (false, None)
    }

    /// Returns the base directory path.
    fn base_path(&self) -> PathBuf;
    /// Returns the metadata path.
    fn metadata_path(&self) -> PathBuf;

    /// Load initial data using columnar format from a file path.
    /// Implementations may use this as a helper for async `load_dense`.
    async fn load_dense_from_file(&self, path: &Path) -> StorageResult<DenseMatrix<f64>>;

    /// Compute the full Lance/parquet file path for a logical filetype.
    fn file_path(&self, key: &str) -> PathBuf;

    // =========
    // ASYNC API
    // =========

    /// Saves a dense matrix. Requires metadata to exist.
    async fn save_dense(
        &self,
        key: &str,
        matrix: &DenseMatrix<f64>,
        md_path: &Path,
    ) -> StorageResult<()>;

    /// Loads a dense matrix from storage.
    async fn load_dense(&self, key: &str) -> StorageResult<DenseMatrix<f64>>;

    /// Saves a sparse matrix. Requires metadata to exist.
    async fn save_sparse(
        &self,
        key: &str,
        matrix: &CsMat<f64>,
        md_path: &Path,
    ) -> StorageResult<()>;

    /// Loads a sparse matrix from storage.
    async fn load_sparse(&self, key: &str) -> StorageResult<CsMat<f64>>;

    /// Saves lambda eigenvalues. Requires metadata to exist.
    async fn save_lambdas(&self, lambdas: &[f64], md_path: &Path) -> StorageResult<()>;

    /// Loads lambda eigenvalues from storage.
    async fn load_lambdas(&self) -> StorageResult<Vec<f64>>;

    /// Initializes storage by saving metadata. Must be called first.
    async fn save_metadata(&self, metadata: &GeneMetadata) -> StorageResult<PathBuf>;

    /// Loads metadata from storage.
    async fn load_metadata(&self) -> StorageResult<GeneMetadata>;

    /// Save vectors that are not lambdas but indices.
    #[allow(dead_code)]
    async fn save_index(&self, key: &str, vector: &[usize], md_path: &Path) -> StorageResult<()>;

    /// save a generic f64 sequence
    async fn save_vector(&self, key: &str, vector: &[f64], md_path: &Path) -> StorageResult<()>;

    /// Save centroid_map (vector of usize mapping items to centroids)
    async fn save_centroid_map(&self, map: &[usize], md_path: &Path) -> StorageResult<()>;

    /// Load centroid_map
    async fn load_centroid_map(&self) -> StorageResult<Vec<usize>>;
    /// Save subcentroid_lambdas (tau values for subcentroids)
    async fn save_subcentroid_lambdas(&self, lambdas: &[f64], md_path: &Path) -> StorageResult<()>;
    /// Load subcentroid_lambdas
    async fn load_subcentroid_lambdas(&self) -> StorageResult<Vec<f64>>;
    /// Save subcentroids (dense matrix)
    async fn save_subcentroids(
        &self,
        subcentroids: &DenseMatrix<f64>,
        md_path: &Path,
    ) -> StorageResult<()>;
    /// Load subcentroids
    async fn load_subcentroids(&self) -> StorageResult<Vec<Vec<f64>>>;

    /// Save item norms (precomputed L2 norms for fast distance computation)
    async fn save_item_norms(&self, item_norms: &[f64], md_path: &Path) -> StorageResult<()>;

    /// Load item norms
    async fn load_item_norms(&self) -> StorageResult<Vec<f64>>;

    /// Save cluster assignments (Vec<Option<usize>>)
    async fn save_cluster_assignments(
        &self,
        assignments: &[Option<usize>],
        md_path: &Path,
    ) -> StorageResult<()>;

    /// Load cluster assignments
    async fn load_cluster_assignments(&self) -> StorageResult<Vec<Option<usize>>>;

    /// Load index or generic usize vector from storage.
    #[allow(dead_code)]
    async fn load_index(&self, key: &str) -> StorageResult<Vec<usize>>;

    async fn load_vector(&self, key: &str) -> StorageResult<Vec<f64>>;

    #[cfg(test)]
    async fn save_dense_to_file(data: &DenseMatrix<f64>, path: &PathBuf) -> StorageResult<()>;
}
