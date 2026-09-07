//! #117-1: clustering artifact I/O, split out of [`StorageBackend`].
//!
//! Centroid maps, subcentroids, item norms and cluster assignments are one
//! consumer's ML-pipeline vocabulary. Keeping them on the base trait coupled
//! a general storage abstraction to that vocabulary and forced every
//! implementor to stub methods it does not support. As an extension
//! sub-trait, only clustering-capable backends implement it (#117-2).

use std::path::Path;

use crate::traits::backend::StorageBackend;
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::StorageResult;

/// Clustering-specific artifact save/load (centroid maps, subcentroid
/// lambdas, subcentroids, item norms, cluster assignments).
///
/// Requirements, not defaults: the on-disk layout is backend-specific.
/// A backend implements this trait only if it supports the clustering
/// pipeline artifacts.
pub trait ClusteringArtifacts: StorageBackend {
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
}
