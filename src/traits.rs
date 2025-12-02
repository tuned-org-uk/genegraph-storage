use log::debug;
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::CsMat;
use std::path::{Path, PathBuf};

use crate::StorageResult;
use crate::metadata::GeneMetadata;

/// Async storage backend for Lance-based graph and embedding data.
///
/// This trait defines the minimal async API required to persist and reload
/// all artifacts used by Javelin:
///
/// - Dense matrices (embeddings, eigenmaps, energy maps)
/// - Sparse matrices in CSR form (e.g. Laplacians, adjacency)
/// - Scalar vectors (eigenvalues, norms, generic f64 sequences)
/// - Index-like vectors (usize mappings and cluster assignments)
/// - Clustering metadata (centroid maps, subcentroids, lambdas)
/// - Global metadata describing the dataset layout and dimensions
///
/// ## Initialization
///
/// Storage must be initialized before saving any data:
///
/// 1. Call `save_metadata()` once to write an initial `*_metadata.json`.
/// 2. Subsequent `save_*` calls validate that metadata exists and is consistent.
/// 3. `exists()` can be used to detect and reuse an existing initialized store.
///
/// Filenames are conventionally:
///
/// ```ignore
/// <base dir>/<instance name or name id>_<key>.lance
/// ```
///
/// ## Async usage
///
/// All I/O functions are async and intended to be called from a Tokio runtime.
/// Implementations (e.g. `LanceStorage`) must not create their own runtimes or
/// block on I/O internally.
///
/// ## High-level flow
///
/// - Dense data:
///   - `save_dense("raw_input", &matrix, md_path)`
///   - `load_dense("raw_input")`
///
/// - Sparse data:
///   - `save_sparse("laplacian", &csr, md_path)`
///   - `load_sparse("laplacian")`
///
/// - Scalars and indices:
///   - `save_lambdas`, `load_lambdas`
///   - `save_vector`, `load_vector`
///   - `save_index`, `load_index`
///   - `save_centroid_map`, `load_centroid_map`
///   - `save_item_norms`, `load_item_norms`
///   - `save_cluster_assignments`, `load_cluster_assignments`
///
/// - Clustering structure:
///   - `save_subcentroids`, `load_subcentroids`
///   - `save_subcentroid_lambdas`, `load_subcentroid_lambdas`
///
/// Implementations are free to choose the on-disk layout as long as they honor
/// these logical keys and round-trip semantics.
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
                if let Some(name) = path.file_name().and_then(|n| n.to_str())
                    && name.ends_with("_metadata.json")
                {
                    debug!("StorageBackend::exists: found metadata file at {:?}", path);
                    return (true, Some(path));
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
